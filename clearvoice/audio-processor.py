from clearvoice import ClearVoice
import os
import argparse
import librosa
import numpy as np
import subprocess
import time

def convert_to_wav_mono(input_file, output_file, target_sr):
    """
    将输入音频文件转换为单声道 WAV 格式，并使用目标采样率 target_sr
    """
    print(f"[convert_to_wav_mono] 准备将 {input_file} 转换为单声道 WAV, 采样率={target_sr}")
    try:
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', input_file,
            '-ac', '1',                  # 单声道
            '-ar', str(target_sr),       # 目标采样率
            output_file
        ]
        print("[convert_to_wav_mono] ffmpeg 命令:", " ".join(ffmpeg_cmd))
        start_time = time.time()
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print(f"[convert_to_wav_mono] FFmpeg 转换失败: {result.stderr.decode()}")
            return False
        
        print(f"[convert_to_wav_mono] 转换成功！耗时 {time.time() - start_time:.2f} 秒")
        return True
    except Exception as e:
        print(f"[convert_to_wav_mono] 转换音频时发生错误: {e}")
        return False

def get_audio_info(file_path):
    """
    获取音频文件的信息，包括采样率、比特率和时长
    """
    if not os.path.isfile(file_path):
        print(f"[get_audio_info] 文件不存在: {file_path}")
        return None

    try:
        import soundfile as sf
        with sf.SoundFile(file_path) as sfile:
            sr = sfile.samplerate
            channels = sfile.channels
            frames = len(sfile)
        duration_s = frames / sr if sr > 0 else 0

        file_size = os.path.getsize(file_path)
        bit_rate = 0
        if duration_s > 0:
            bit_rate = int(round((file_size * 8) / duration_s / 1000))

        return {
            'sample_rate': sr,
            'bit_rate': bit_rate,
            'duration': duration_s,
            'channels': channels
        }
    except Exception as e:
        print(f"[get_audio_info] 使用 soundfile 读取失败，尝试 librosa.load: {e}")

    # 如果 soundfile 失败，fallback 到 librosa.load
    try:
        import librosa
        y, sr = librosa.load(file_path, sr=None, mono=False)
        duration_s = librosa.get_duration(y=y, sr=sr)
        file_size = os.path.getsize(file_path)
        if duration_s > 0:
            bit_rate = int(round((file_size * 8) / duration_s / 1000))
        else:
            bit_rate = 0

        channels = 1 if y.ndim == 1 else y.shape[0]
        return {
            'sample_rate': sr,
            'bit_rate': bit_rate,
            'duration': duration_s,
            'channels': channels
        }
    except Exception as e2:
        print(f"[get_audio_info] 再次读取失败: {e2}")
        return None

def save_wav(wav_data, sr, output_path):
    """
    使用 ffmpeg 将处理后的音频数据（float32，单声道）保存为 .wav 格式 (16bit PCM)。
    确保输出采样率与指定的 sr 一致。
    """
    print(f"[save_wav] 准备将增强后的数据写入 {output_path} (采样率={sr})")
    try:
        # 确保音频数据在 [-1,1]，类型为 float32
        wav_data = np.array(wav_data, dtype=np.float32)
        wav_data = np.clip(wav_data, -1.0, 1.0)

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'f32le',       # 输入数据格式：32位浮点
            '-ar', str(sr),      # 输入采样率
            '-ac', '1',          # 输入1声道
            '-i', 'pipe:0',      # 从 pipe 读取
            '-c:a', 'pcm_s16le', # 输出 WAV (16-bit PCM)
            '-ar', str(sr),      # 输出采样率
            '-af', 'aresample=resampler=soxr',  # 使用高质量重采样器
            output_path
        ]
        print("[save_wav] ffmpeg 命令:", " ".join(ffmpeg_cmd))
        start_time = time.time()
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        process.stdin.write(wav_data.tobytes())
        process.stdin.close()

        stderr = process.stderr.read()
        process.stderr.close()
        process.wait()

        if process.returncode != 0:
            print(f"[save_wav] FFmpeg 警告: {stderr.decode()}")
            return False

        print(f"[save_wav] 保存成功！耗时 {time.time() - start_time:.2f} 秒")
        return True
    except Exception as e:
        print(f"[save_wav] 保存 WAV 文件时发生错误: {e}")
        return False

def process_audio(input_file, output_dir=None):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件未找到: {input_file}")

    print("[process_audio] Step1: 分析输入音频")
    audio_info = get_audio_info(input_file)
    if not audio_info:
        raise ValueError("无法分析输入音频文件。")

    print(f"[process_audio] 输入音频信息:\n"
          f"  采样率: {audio_info['sample_rate']} Hz\n"
          f"  通道数: {audio_info['channels']}\n"
          f"  比特率: {audio_info['bit_rate']} kbps\n"
          f"  时长  : {audio_info['duration']:.2f} 秒")

    # 选择模型
    if audio_info['sample_rate'] >= 44100:
        enhance_model = 'MossFormer2_SE_48K'
        sr_model = 'MossFormer2_SR_48K'
        model_sr = 48000
    else:
        enhance_model = 'FRCRN_SE_16K'
        sr_model = None  # 16K没有SR模型
        model_sr = 16000

    print(f"\n[process_audio] Step2: 准备模型")
    print(f"  选择降噪模型: {enhance_model} (采样率={model_sr}Hz)")
    if sr_model:
        print(f"  选择音质提升模型: {sr_model}")

    # 设置输出路径
    if output_dir is None:
        output_dir = os.path.dirname(input_file) or '.'
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.wav")
    final_path = os.path.join(output_dir, f"{base_name}_final.wav")

    # 格式转换
    if (audio_info['channels'] != 1 or 
        audio_info['sample_rate'] != model_sr or 
        not input_file.lower().endswith('.wav')):
        from tempfile import gettempdir
        model_input_path = os.path.join(gettempdir(), f"model_input_{os.path.basename(input_file)}.wav")
        print(f"  转换音频格式: {audio_info['channels']}ch@{audio_info['sample_rate']}Hz -> 1ch@{model_sr}Hz")
        if not convert_to_wav_mono(input_file, model_input_path, model_sr):
            raise RuntimeError("音频格式转换失败。")
    else:
        model_input_path = input_file
        print("  输入音频格式符合要求，无需转换")

    # 降噪处理
    print("\n[process_audio] Step3: 执行降噪")
    cv_enhance = ClearVoice(task='speech_enhancement', model_names=[enhance_model])
    start_time = time.time()
    enhanced_data = cv_enhance(input_path=model_input_path, online_write=False)
    print(f"  降噪完成，耗时 {time.time() - start_time:.2f} 秒")

    if isinstance(enhanced_data, dict):
        enhanced_wav = list(enhanced_data.values())[0]
    else:
        enhanced_wav = enhanced_data
    enhanced_wav = np.array(enhanced_wav)
    
    # 保存降噪结果
    if not save_wav(enhanced_wav, model_sr, enhanced_path):
        print("\n[process_audio] 保存降噪文件失败")
        return None

    # 音质提升处理
    if sr_model:
        print("\n[process_audio] Step4: 执行音质提升")
        cv_sr = ClearVoice(task='speech_super_resolution', model_names=[sr_model])
        start_time = time.time()
        sr_data = cv_sr(input_path=enhanced_path, online_write=False)
        print(f"  音质提升完成，耗时 {time.time() - start_time:.2f} 秒")

        if isinstance(sr_data, dict):
            sr_wav = list(sr_data.values())[0]
        else:
            sr_wav = sr_data
        sr_wav = np.array(sr_wav)
        
        # 保存最终结果
        if save_wav(sr_wav, model_sr, final_path):
            result_info = get_audio_info(final_path)
            print("\n[process_audio] 处理完成！")
            print(f"输出文件: {final_path}")
            if result_info:
                print(f"输出音频信息:\n"
                      f"  采样率: {result_info['sample_rate']} Hz\n"
                      f"  通道数: {result_info['channels']}\n"
                      f"  时长  : {result_info['duration']:.2f} 秒")
        else:
            print("\n[process_audio] 保存最终文件失败")
            return enhanced_path
        return final_path
    else:
        print("\n[process_audio] 处理完成！(采样率较低，跳过音质提升)")
        return enhanced_path

def main():
    parser = argparse.ArgumentParser(description='音频增强处理工具')
    parser.add_argument('input', help='输入音频文件路径')
    parser.add_argument('--output-dir', help='输出目录（可选）')
    # 只保留必要的参数
    args = parser.parse_args()

    try:
        process_audio(
            input_file=args.input,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"[main] 处理文件时发生错误: {e}")
        exit(1)
    exit(0)

if __name__ == "__main__":
    main()