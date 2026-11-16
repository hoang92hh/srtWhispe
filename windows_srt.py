"""Giao diện Windows tạo SRT tiếng Trung từ các tệp WAV."""

import argparse
import ctypes
import gc
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


COMPUTE_TYPE = "int8_float32"
MODELS = ("large-v2", "medium", "small", "base", "tiny")


def prepare_cuda():
    if sys.platform != "win32":
        raise RuntimeError("Hãy chạy script này trên Windows.")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"PyTorch {torch.__version__} chưa nhận GPU CUDA. "
            "Hãy kiểm tra nvidia-smi và cài PyTorch cu126 theo hướng dẫn."
        )

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    for dll in ("cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"):
        ctypes.WinDLL(str(torch_lib / dll))

    import ctranslate2

    torch.ones(1, device="cuda").item()
    if COMPUTE_TYPE not in ctranslate2.get_supported_compute_types("cuda"):
        raise RuntimeError(f"CTranslate2 chưa hỗ trợ {COMPUTE_TYPE} trên GPU đang dùng.")
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__} | {COMPUTE_TYPE}")


def process_wavs(wavs, output_folder, run_srt, *, model_name, overwrite=False, progress=print):
    diagnostics = output_folder / "diagnostics"
    saved_count = 0
    failed_count = 0
    review_count = 0
    for index, wav in enumerate(wavs, 1):
        srt = output_folder / f"{wav.stem}.srt"
        progress(f"[{index}/{len(wavs)}] {wav.name}")
        try:
            result = run_srt(
                wav, diagnostics, model_name=model_name, compute_type=COMPUTE_TYPE,
                srt_destination=srt, overwrite_srt=overwrite,
            )
            saved_count += 1
            review_count += int(result["review_required"])
            progress(f"Đã lưu {srt}")
            del result
        except (OSError, RuntimeError, ValueError) as error:
            progress(f"Lỗi khi xử lý {wav.name}: {error}")
            failed_count += 1
        finally:
            gc.collect()

    progress(f"Hoàn tất. Đã lưu {saved_count} SRT trong {output_folder}.")
    if review_count:
        progress(f"{review_count} tệp cần nghe kiểm tra. Xem report.json trong {diagnostics}.")
    if failed_count:
        progress(f"{failed_count}/{len(wavs)} WAV xử lý lỗi.")
    return 1 if failed_count else 0


class SrtApp:
    def __init__(self, root, initial_model):
        self.root = root
        self.events = queue.Queue()
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.model = tk.StringVar(value=initial_model)
        self.status = tk.StringVar(value="Chọn thư mục và model để bắt đầu.")
        root.title("Tạo SRT từ WAV")
        root.minsize(660, 240)
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        for row, (label, variable, command) in enumerate((
            ("Thư mục WAV", self.input_folder, self.choose_input),
            ("Thư mục kết quả", self.output_folder, self.choose_output),
        )):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=6)
            ttk.Entry(frame, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8)
            ttk.Button(frame, text="Chọn...", command=command).grid(row=row, column=2)

        ttk.Label(frame, text="Model Whisper").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Combobox(frame, textvariable=self.model, values=MODELS, state="readonly", width=18).grid(
            row=2, column=1, sticky="w", padx=8
        )
        self.start_button = ttk.Button(frame, text="Tạo srt file", command=self.start)
        self.start_button.grid(row=3, column=1, sticky="w", padx=8, pady=12)
        ttk.Label(frame, textvariable=self.status, wraplength=600).grid(
            row=4, column=0, columnspan=3, sticky="w"
        )
        root.after(100, self.poll_events)

    def choose_input(self):
        selected = filedialog.askdirectory(title="Chọn thư mục WAV", parent=self.root, mustexist=True)
        if selected:
            self.input_folder.set(selected)

    def choose_output(self):
        selected = filedialog.askdirectory(title="Chọn thư mục kết quả", parent=self.root, mustexist=True)
        if selected:
            self.output_folder.set(selected)

    def start(self):
        input_folder = Path(self.input_folder.get())
        output_folder = Path(self.output_folder.get())
        if not input_folder.is_dir() or not output_folder.is_dir():
            messagebox.showerror("Thiếu thư mục", "Hãy chọn thư mục WAV và thư mục kết quả hợp lệ.", parent=self.root)
            return
        wavs = sorted(
            (path for path in input_folder.iterdir() if path.is_file() and path.suffix.lower() == ".wav"),
            key=lambda path: path.name.casefold(),
        )
        if not wavs:
            messagebox.showerror("Không có WAV", "Thư mục đầu vào không có tệp WAV.", parent=self.root)
            return
        existing = [wav for wav in wavs if (output_folder / f"{wav.stem}.srt").exists()]
        overwrite = False
        if existing:
            answer = messagebox.askyesnocancel(
                "SRT đã tồn tại",
                f"Có {len(existing)} SRT trùng tên. Ghi đè các tệp này?\n"
                "Yes: ghi đè. No: bỏ qua. Cancel: dừng.",
                parent=self.root,
            )
            if answer is None:
                return
            overwrite = answer
            if not overwrite:
                wavs = [wav for wav in wavs if wav not in existing]
        if not wavs:
            self.status.set("Không có WAV nào cần xử lý.")
            return
        self.start_button.configure(state="disabled")
        self.status.set(f"Đang chạy {self.model.get()} với {len(wavs)} WAV...")
        threading.Thread(
            target=self.run, args=(wavs, output_folder, self.model.get(), overwrite), daemon=True
        ).start()

    def run(self, wavs, output_folder, model_name, overwrite):
        try:
            prepare_cuda()
            from colab_srt_t4 import run_vad_stable_srt

            result = process_wavs(
                wavs, output_folder, run_vad_stable_srt, model_name=model_name,
                overwrite=overwrite, progress=lambda message: self.events.put(("progress", message)),
            )
            self.events.put(("done", result))
        except Exception as error:
            self.events.put(("error", f"{type(error).__name__}: {error}"))

    def poll_events(self):
        while not self.events.empty():
            kind, value = self.events.get_nowait()
            if kind == "progress":
                self.status.set(value)
                print(value, flush=True)
            else:
                self.start_button.configure(state="normal")
                if kind == "error":
                    self.status.set(value)
                    messagebox.showerror("Lỗi tạo SRT", value, parent=self.root)
                elif value == 0:
                    messagebox.showinfo("Hoàn tất", self.status.get(), parent=self.root)
        self.root.after(100, self.poll_events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="large-v2")
    parser.add_argument("--check-gpu", action="store_true")
    args = parser.parse_args()
    if args.check_gpu:
        prepare_cuda()
        from colab_srt_t4 import run_vad_stable_srt

        print("Đã kiểm tra CUDA, CTranslate2 và import thư viện của script.")
        return 0
    root = tk.Tk()
    SrtApp(root, args.model)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    raise SystemExit(main())
