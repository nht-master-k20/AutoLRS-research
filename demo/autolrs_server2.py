import sys

try:
    import joblib

    sys.modules['sklearn.externals.joblib'] = joblib
except ImportError:
    pass

import matplotlib

matplotlib.use('Agg')
import traceback
import argparse
import socket
import numpy as np
import threading
import math
import os
from skopt import Optimizer
from skopt.space import Real
from scipy.interpolate import UnivariateSpline
from scipy import optimize
import logging

logging.basicConfig(level=logging.INFO)


# --- Helper functions (Fix lỗi dfitpack crash) ---
def f(b, x, y):
    A = np.vstack([np.exp(-np.exp(b) * x), np.ones(len(x))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    residuals = result[1]
    if residuals.size == 0: return 0.0
    return residuals.item()


def spline_iter(xs, ys, is_training, spline_deg=2, filter_ratio=0.03, num_of_iter=10, bound=0.5):
    if len(xs) <= spline_deg + 1: return xs, ys
    try:
        bound_idx = int((len(xs) - 1) * bound)
        bound_val = xs[bound_idx]
    except:
        return xs, ys

    if is_training:
        num_of_iter = 10
    else:
        num_of_iter = 1

    for _ in range(num_of_iter):
        if len(xs) <= spline_deg + 1: break
        try:
            spline_ys = UnivariateSpline(xs, ys, k=spline_deg)(xs)
        except Exception:
            break
        dys = np.abs(ys - spline_ys)
        if is_training:
            num_outliers = int(round(len(dys) * filter_ratio))
            if num_outliers < 1: num_outliers = 1
            outliers = set(sorted(range(len(dys)), key=lambda i: dys[i])[-num_outliers:])
        else:
            outliers = set(sorted(range(len(dys)), key=lambda i: dys[i])[-1:])

        real_outliers = []
        for i in outliers:
            if i < len(xs) and xs[i] < bound_val: real_outliers.append(i)
        if not real_outliers: break
        mask = np.ones(len(xs), dtype=bool)
        for i in real_outliers: mask[i] = False
        xs, ys = xs[mask], ys[mask]
    return xs, ys


def exp_forecast(loss_series, end_step, is_training, spline_order=2):
    xs = np.arange(end_step - len(loss_series), end_step)
    xs2, ys2 = spline_iter(xs, loss_series, is_training)
    if len(xs2) <= spline_order:
        ys = ys2
    else:
        try:
            ys = UnivariateSpline(xs2, ys2, k=spline_order)(xs)
        except:
            ys = np.interp(xs, xs2, ys2)
    try:
        b = optimize.fmin(f, 0, args=(xs, ys), xtol=1e-5, ftol=1e-5, disp=False)[0]
        b = -np.exp(b)
        A = np.vstack([np.exp(b * xs), np.ones(len(xs))]).T
        a, c = np.linalg.lstsq(A, ys, rcond=None)[0]
    except:
        return 0, 0, ys[-1] if len(ys) > 0 else 0
    return a, b, c


class RingBuffer:
    def __init__(self, size): self.data = [None for i in range(size)]

    def reset(self): self.data = [None for i in self.data]

    def append(self, x):
        self.data.pop(0)
        self.data.append(x)

    def get(self): return self.data

    def exponential_forcast(self, pred_index, is_training):
        loss_series = [x for x in self.data if x is not None]
        if len(loss_series) < 3: return loss_series[-1] if loss_series else 0.0
        y = np.array(loss_series)
        a3, b3, c3 = exp_forecast(y, len(y), is_training)
        return a3 * np.exp(b3 * pred_index) + c3


class Controller(object):
    def __init__(self, host, port, min_lr, max_lr):
        # --- CẤU HÌNH CHUẨN PAPER ---
        self.INITIAL_EXPLOITATION_STEP = 1000  #
        self.INITIAL_LR_STEPS = 100  #
        self.TAU_MAX = 8000  # [cite: 260]
        LR_TO_EXPLORE = 10  # [cite: 259]

        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.host = host
        self.port = port
        self.threads = []
        self.num_threads = 1
        self.event = threading.Event()
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))

        self.global_step = 0
        self.loss_vector = []
        self.lr = 0

        # Init Curriculum
        self.exploitation_step = self.INITIAL_EXPLOITATION_STEP
        self.lr_steps = self.INITIAL_LR_STEPS
        self.ring_buffer_len = self.lr_steps
        self.lr_to_explore = LR_TO_EXPLORE

        self.val_freq = max(1, int(self.lr_steps / 16))
        self.lr_counter = 0
        self.BO_stage = True
        self.val_stage = False
        self.exploitation_flag = False
        self.exploitation_counter = 0
        self.loss_after_exploitation = None
        self.average_loss = 0.0
        self.init_loss = 0.0

        self.ring_loss_buffer = RingBuffer(self.ring_buffer_len)
        self.opt = None
        self.x_func_dict = dict()
        self.x_iters = []
        self.func_val_iters = []

        self.num_ranks = 0
        self.finished_minions = 0
        self.lock1 = threading.Lock()
        self.lock2 = threading.Lock()

    def listen(self):
        self.sock.listen(20)
        logging.info(f'[Server] Listening on {self.port}...')
        while True:
            client, address = self.sock.accept()
            self.threads.append(threading.Thread(target=self.run, args=(client, address, self.event)))
            if len(self.threads) == self.num_threads:
                self.num_ranks = len(self.threads)
                self.num_minions = self.num_ranks - 1
                for thread in self.threads: thread.start()
                for thread in self.threads: thread.join()
                self.threads = []

    def run(self, c, address, event):
        size = 1024
        try:
            while True:
                data = c.recv(size).decode()
                if not data: return
                total_loss = float(data.split(',')[-1])
                self.lock2.acquire()
                self.loss_vector.append(total_loss)

                if len(self.loss_vector) == self.num_ranks:
                    self.average_loss = sum(self.loss_vector) / len(self.loss_vector)
                    logging.info(f'[Step {self.global_step}] Avg Loss: {self.average_loss:.4f}')
                    if self.val_stage:
                        if 'val' in data:
                            self.ring_loss_buffer.append(self.average_loss)
                        else:
                            self.global_step += 1
                    else:
                        self.ring_loss_buffer.append(self.average_loss)
                        self.global_step += 1
                    self.loss_vector = []
                self.lock2.release()

                if 'val' in data:
                    c.send(str(self.lr).encode('utf-8'))
                    continue
                if 'minion' in data:
                    event.wait()
                    c.send(self.message.encode('utf-8'))
                    self.lock1.acquire()
                    self.finished_minions += 1
                    self.lock1.release()
                    if self.finished_minions == self.num_minions:
                        event.clear()
                        self.finished_minions = 0
                    continue

                if data.startswith('startBO'):
                    self.loss_after_exploitation = self.average_loss
                    self.init_loss = self.average_loss

                # --- LOGIC EXPLOITATION & CURRICULUM ---
                if self.exploitation_flag:
                    if self.exploitation_counter == self.exploitation_step:
                        self.BO_stage = True
                        self.exploitation_flag = False
                        self.exploitation_counter = 0
                        logging.info('[Server] Exploitation Done. Reconfiguring...')

                        # [IMPLEMENT PAPER LOGIC: Double tau until tau_max]
                        if self.exploitation_step < self.TAU_MAX:
                            self.exploitation_step *= 2
                            self.lr_steps *= 2
                            self.ring_buffer_len = self.lr_steps
                            logging.info(f'>>> CURRICULUM LEVEL UP: New tau={self.exploitation_step}')

                        self.val_freq = max(1, int(self.lr_steps / 16))
                        if self.val_stage:
                            self.ring_loss_buffer = RingBuffer(self.ring_buffer_len // self.val_freq)
                        else:
                            self.ring_loss_buffer = RingBuffer(self.ring_buffer_len)

                        self.loss_after_exploitation = self.average_loss
                        self.message = 'save'
                        c.send(self.message.encode('utf-8'))
                        event.set()
                        continue
                    else:
                        self.exploitation_counter += 1
                        self.message = str(self.lr)
                        c.send(str(self.lr).encode('utf-8'))
                        event.set()
                        continue

                # --- LOGIC BO ---
                if self.BO_stage:
                    self.opt = Optimizer([Real(self.min_lr, self.max_lr, 'log-uniform')], "GP",
                                         n_initial_points=1, acq_func='LCB', acq_func_kwargs={'kappa': 1000})
                    self.BO_stage = False
                    self.lr = self.opt.ask()[0]
                    while self.lr in self.x_func_dict: self.lr = self.opt.ask()[0]
                    self.message = f'ckpt,{self.lr}'
                    c.send(self.message.encode('utf-8'))
                    event.set()
                    continue

                if self.lr_counter == self.lr_steps:
                    buffer_data = self.ring_loss_buffer.get()
                    has_bad_data = any(x is None or math.isnan(x) for x in buffer_data if isinstance(x, (int, float)))
                    if has_bad_data:
                        predicted_loss = "nan"
                    elif self.val_stage:
                        predicted_loss = self.ring_loss_buffer.exponential_forcast(
                            pred_index=int(self.exploitation_step / self.val_freq), is_training=False)
                    else:
                        predicted_loss = self.ring_loss_buffer.exponential_forcast(
                            pred_index=self.exploitation_step, is_training=True)

                    logging.info(f'[Server] LR {self.lr:.6f} -> Pred Loss: {predicted_loss}')

                    if str(predicted_loss) == 'nan':
                        self.opt.tell([float(self.lr)], 1e6)
                    else:
                        self.opt.tell([float(self.lr)], predicted_loss)

                    self.x_iters.append(float(self.lr))
                    self.func_val_iters.append(predicted_loss)
                    self.x_func_dict[self.lr] = predicted_loss
                    self.lr_counter = 1

                    if self.val_stage:
                        self.ring_loss_buffer = RingBuffer(int(math.floor(self.ring_buffer_len) / self.val_freq))
                    else:
                        self.ring_loss_buffer = RingBuffer(self.ring_buffer_len)

                    if len(self.func_val_iters) == self.lr_to_explore:
                        best_idx = self.func_val_iters.index(min(self.func_val_iters))
                        self.lr = self.x_iters[best_idx]
                        self.message = f'restore,{self.lr}'
                        c.send(self.message.encode('utf-8'))
                        event.set()
                        self.exploitation_flag = True
                        self.func_val_iters = []
                        self.x_iters = []
                        self.x_func_dict = dict()
                    else:
                        self.lr = self.opt.ask()[0]
                        while self.lr in self.x_func_dict: self.lr = self.opt.ask()[0]
                        self.message = f'restore,{self.lr}'
                        c.send(self.message.encode('utf-8'))
                        event.set()
                else:
                    self.lr_counter += 1
                    if self.val_stage and self.lr_counter % self.val_freq == 0:
                        self.message = "evaluate"
                    else:
                        self.message = str(self.lr)
                    c.send(self.message.encode('utf-8'))
                    event.set()

        except Exception:
            traceback.print_exc()
            try:
                c.send(str(self.lr).encode('utf-8'))
            except:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_lr', required=True)
    parser.add_argument('--max_lr', required=True)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=12315)
    args = parser.parse_args()
    Controller(args.host, args.port, args.min_lr, args.max_lr).listen()


if __name__ == '__main__': main()