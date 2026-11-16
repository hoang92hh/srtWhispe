# Cài runtime Windows để tạo SRT từ WAV

`windows_srt.py` mở cửa sổ để chọn thư mục WAV, thư mục lưu kết quả và model Whisper. Script dùng quy trình từ `colab_srt_t4.py` để nhận dạng tiếng Trung, chia đoạn SRT, rà vùng tiếng nói còn thiếu và tinh chỉnh mốc bắt đầu. Giữ hai tệp `.py` trong cùng một thư mục.

## 1. Cài thư viện bằng PowerShell

Máy đã có Python 3.13 x64, Tkinter và driver NVIDIA 560.94. Các lệnh dùng lại `.venv` trong `C:\tool` nếu đã có. Bản PyTorch `cu126` dùng CUDA 12.6. Script chọn `int8_float32` cho GTX 1060 6 GB, thuộc nhóm compute capability 6.1. [PyTorch](https://docs.pytorch.org/get-started/previous-versions/) · [Danh sách GPU NVIDIA](https://developer.nvidia.com/cuda/gpus/legacy) · [Kiểu tính toán CTranslate2](https://opennmt.net/CTranslate2/quantization.html)

```powershell
Set-Location 'C:\tool'
$python = '.\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    python -m venv .venv
}

$gpuPackages = @('torch==2.8.0+cu126', 'torchaudio==2.8.0+cu126')
& $python -m pip install --force-reinstall @gpuPackages --index-url 'https://download.pytorch.org/whl/cu126'

$packages = @(
    'stable-ts[fw]==2.19.1'
    'faster-whisper==1.2.1'
    'silero-vad==6.2.3'
    'ctranslate2==4.6.1'
    'setuptools<81'
    'av<19'
    'numpy'
    'scipy'
    'soundfile'
)
& $python -m pip install @gpuPackages @packages
& $python -m pip install 'pytorch-metric-learning<3'
& $python -m pip check
```

`pip check` cần trả về `No broken requirements found.` CTranslate2 4.6.1 vẫn dùng `pkg_resources` trên Windows, nên lệnh cài giới hạn phiên bản `setuptools`. Script nạp các DLL CUDA và cuDNN từ bản PyTorch trong `.venv`. [Bản build CTranslate2 4.6.1 cho Windows](https://github.com/OpenNMT/CTranslate2/blob/v4.6.1/python/tools/prepare_build_environment_windows.sh) · [Mã nạp DLL](https://github.com/OpenNMT/CTranslate2/blob/v4.6.1/python/ctranslate2/__init__.py)

## 2. Kiểm tra GPU

```powershell
Set-Location 'C:\tool'
& .\.venv\Scripts\python.exe .\windows_srt.py --check-gpu
```

Lệnh cần in tên GTX 1060, phiên bản PyTorch `2.8.0+cu126` và `int8_float32`. Bạn có thể chạy bước này trước khi tải model nhận dạng.

## 3. Chọn thư mục và chạy

```powershell
Set-Location 'C:\tool'
& .\.venv\Scripts\python.exe -u .\windows_srt.py
```

1. Chọn thư mục chứa WAV và thư mục lưu kết quả trong cửa sổ. Script đọc các tệp `.wav` nằm trực tiếp trong thư mục, theo thứ tự tên.
2. Chọn model Whisper trong hộp chọn. Mặc định là `large-v2`.
3. Bấm **Tạo srt file** và theo dõi tiến độ trên cửa sổ hoặc PowerShell. Lần đầu Faster-Whisper sẽ tải model đã chọn qua Internet. [Danh sách model Faster-Whisper 1.2.1](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/utils.py)

SRT được lưu trực tiếp trong thư mục đầu ra với tên tương ứng, ví dụ `mau.wav` tạo `mau.srt`. Nếu đã có SRT trùng tên, chọn **Yes** để ghi đè hoặc **No** để giữ các tệp cũ và tiếp tục với những WAV còn lại.

Báo cáo JSON của từng WAV được lưu trong thư mục con `diagnostics/`. Nếu một WAV gặp lỗi, PowerShell hiển thị tên tệp và tiếp tục xử lý các WAV còn lại.

Nếu gặp `CUDA out of memory`, thử model `medium` để giảm VRAM. Model này vẫn hỗ trợ tiếng Trung.

```powershell
& .\.venv\Scripts\python.exe -u .\windows_srt.py --model medium
```

Nếu Python báo thiếu gói, chạy lại bước 1. Nếu `nvidia-smi` nhận GPU nhưng script báo `DLL load failed`, kiểm tra bản PyTorch `cu126`. Với lỗi thiếu `VCRUNTIME` hoặc `MSVCP`, kiểm tra [Microsoft Visual C++ Redistributable x64](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
