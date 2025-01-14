from clearvoice import ClearVoice
import os
import argparse
import librosa
import numpy as np
import subprocess

def convert_to_wav_mono(input_file, output_file, target_sr):
    """
    将输入音频文件转换为单声道 WAV 格式，并使用原音频的采样率。

    参数:
    ----------
    input_file (str): 原始音频文件路径。
    output_file (str): 转换后的 WAV 文件路径。
    target_sr (int): 目标采样率，通常取原音频文件的采样率。
    
    返回:
    -------
    bool: 如果转换成功则返回 True，否则返回 False。
    """
    try:
        # 使用 ffmpeg 将音频转换为单声道 WAV，采样率与输入文件一致
        ffmpeg_cmd = [
            'ffmpeg', '-y',              # 覆盖输出文件
            '-i', input_file,            # 输入文件
            '-ac', '1',                  # 设置为单声道
            '-ar', str(target_sr),       # 使用原音频文件的采样率
            output_file                  # 输出文件
        ]
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"FFmpeg 转换失败: {result.stderr.decode()}")
            return False
        return True
    except Exception as e:
        print(f"转换音频时发生错误: {e}")
        return False

def get_audio_info(file_path):
    """
    获取音频文件的信息，包括采样率、比特率和时长。

    参数:
    ----------
    file_path (str): 音频文件路径。
    
    返回:
    -------
    dict: 包含采样率（sample_rate）、比特率（bit_rate）和时长（duration）的字典。
    """
    try:
        # librosa.load 默认读取整个音频，并返回音频数据 y 和采样率 sr
        y, sr = librosa.load(file_path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        
        file_size = os.path.getsize(file_path)
        if duration > 0:
            bit_rate = int(round((file_size * 8) / duration / 1000))
        else:
            bit_rate = 0
        
        return {
            'sample_rate': sr,
            'bit_rate': bit_rate,
            'duration': duration
        }
    except Exception as e:
        print(f"警告: 无法获取音频信息: {e}")
        return None

def save_audio(wav_data, sr, output_path, target_bit_rate):
    """
    将音频数据保存为 MP3 文件，指定比特率。

    参数:
    ----------
    wav_data (numpy.ndarray): 音频数据。
    sr (int): 采样率。
    output_path (str): 输出文件路径。
    target_bit_rate (int): 目标比特率（kbps）。
    
    返回:
    -------
    bool: 如果保存成功则返回 True，否则返回 False。
    """
    try:
        # 确保音频数据为 float32 并在 [-1, 1] 范围内
        wav_data = np.array(wav_data, dtype=np.float32)
        wav_data = np.clip(wav_data, -1, 1)
        
        # 创建 ffmpeg 命令
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'f32le',           # 输入格式（32位浮点）
            '-ar', str(sr),          # 采样率
            '-ac', '1',              # 单声道
            '-i', 'pipe:0',          # 从标准输入读取
            '-c:a', 'libmp3lame',    # MP3 编码器
            '-b:a', f'{target_bit_rate}k',  # 目标比特率
            '-ar', str(sr),          # 输出采样率
            output_path
        ]
        
        # 启动 ffmpeg 进程
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 将音频数据写入 ffmpeg 的标准输入
        process.stdin.write(wav_data.tobytes())
        process.stdin.close()
        
        # 等待进程完成
        stderr = process.stderr.read()
        process.stderr.close()
        process.wait()
        
        if process.returncode != 0:
            print(f"FFmpeg 警告: {stderr.decode()}")
            return False
        return True
    except Exception as e:
        print(f"保存音频时发生错误: {e}")
        return False

def process_audio(input_file, output_dir=None, target_bit_rate=None, preserve_quality=True):
    """
    使用 ClearVoice 对音频文件进行增强处理。

    参数:
    ----------
    input_file (str): 输入音频文件路径。
    output_dir (str, optional): 输出目录。如果为 None，则使用输入文件所在目录。
    target_bit_rate (int, optional): 目标比特率（kbps）。如果为 None，根据是否保留质量自动选择。
    preserve_quality (bool): 是否保留或提升原始质量。
    
    返回:
    -------
    str: 增强后的音频文件路径。
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件未找到: {input_file}")
    
    # 先获取输入文件（可能是mp3、aac等）的信息
    audio_info = get_audio_info(input_file)
    if not audio_info:
        raise ValueError("无法分析输入音频文件。")
    
    original_sr = audio_info['sample_rate']
    print(f"原始音频信息:")
    print(f"采样率: {audio_info['sample_rate']} Hz")
    print(f"比特率: {audio_info['bit_rate']} kbps")
    print(f"时长: {audio_info['duration']:.2f} 秒")

    # 设置转换后的临时 WAV 文件路径
    base, ext = os.path.splitext(input_file)
    wav_mono_path = base + '_mono.wav'

    # 如果输入文件不是 WAV，或其通道数不是 1（单声道），则转换
    # 此处不检测声道数（如想更严谨，可扩展 get_audio_info 来获取声道数），
    # 简化起见，若不是 WAV 就先统一转换。
    if ext.lower() != '.wav':
        print("正在将输入音频转换为单声道 WAV 格式...")
        success = convert_to_wav_mono(input_file, wav_mono_path, original_sr)
        if not success:
            raise ValueError("音频格式转换失败。请检查输入文件。")
    else:
        # 如果已经是 wav 格式，也要确保是单声道、并且采样率和原始一致，
        # 这里直接转换一次以简化逻辑（根据需求也可先判断是否真的需要转换）
        print("已是 WAV 格式，但仍将检查/转换为单声道。")
        success = convert_to_wav_mono(input_file, wav_mono_path, original_sr)
        if not success:
            raise ValueError("音频格式转换失败。请检查输入文件。")

    # 读取新的单声道 WAV 文件的信息
    mono_wav_info = get_audio_info(wav_mono_path)
    if not mono_wav_info:
        raise ValueError("无法分析转换后的 WAV 文件。")
    
    # 再次更新 original_sr，确保与转换后的文件保持一致
    original_sr = mono_wav_info['sample_rate']
    print(f"\n转换后（单声道 WAV）的音频信息:")
    print(f"采样率: {mono_wav_info['sample_rate']} Hz")
    print(f"比特率: {mono_wav_info['bit_rate']} kbps")
    print(f"时长: {mono_wav_info['duration']:.2f} 秒")

    # 确定目标比特率
    if target_bit_rate is None:
        if preserve_quality:
            target_bit_rate = max(mono_wav_info['bit_rate'], 192)
        else:
            target_bit_rate = min(mono_wav_info['bit_rate'], 192)
    
    # 根据采样率和比特率选择适当的模型
    if mono_wav_info['sample_rate'] >= 44100 or target_bit_rate >= 192:
        enhance_model = 'MossFormer2_SE_48K'
    else:
        enhance_model = 'FRCRN_SE_16K'
    
    print(f"\n使用增强模型: {enhance_model}")
    print(f"目标比特率: {target_bit_rate} kbps")
    
    # 设置输出路径
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.mp3")
    
    # 初始化 ClearVoice 模型
    print("\n初始化语音增强模型...")
    cv_enhance = ClearVoice(task='speech_enhancement', model_names=[enhance_model])
    
    # 处理音频
    print(f"正在增强音频: {os.path.basename(wav_mono_path)}")
    enhanced_wav = cv_enhance(input_path=wav_mono_path, online_write=False)
    
    # 确定模型的采样率
    model_sr = 48000 if enhance_model == 'MossFormer2_SE_48K' else 16000
    
    # 确保增强后的音频为 numpy 数组
    enhanced_wav = np.array(enhanced_wav)
    
    # 计算预期的样本数
    expected_length = int(len(enhanced_wav) * original_sr / model_sr)
    
    print(f"\n调试信息:")
    print(f"原始采样率: {original_sr} Hz")
    print(f"模型采样率: {model_sr} Hz")
    print(f"增强后音频样本数: {len(enhanced_wav)}")
    print(f"重采样后的预期样本数: {expected_length}")
    
    # 重采样以匹配原始采样率
    if model_sr != original_sr:
        enhanced_wav = librosa.resample(
            enhanced_wav, 
            orig_sr=model_sr,
            target_sr=original_sr
        )
    
    # 验证重采样后的样本数
    actual_length = len(enhanced_wav)
    print(f"重采样后的实际样本数: {actual_length}")
    print(f"预期时长: {expected_length / original_sr:.2f} 秒")
    print(f"实际时长: {actual_length / original_sr:.2f} 秒")
    
    # 保存增强后的音频
    if save_audio(enhanced_wav, original_sr, enhanced_path, target_bit_rate):
        print(f"\n处理完成！")
        print(f"增强后的文件已保存至: {enhanced_path}")
        
        # 验证最终输出
        final_info = get_audio_info(enhanced_path)
        if final_info:
            print(f"\n最终音频信息:")
            print(f"采样率: {final_info['sample_rate']} Hz")
            print(f"比特率: {final_info['bit_rate']} kbps")
            print(f"时长: {final_info['duration']:.2f} 秒")
    else:
        print("\n保存增强后的音频失败。")
    
    return enhanced_path

def main():
    parser = argparse.ArgumentParser(description='使用 ClearVoice 处理音频文件')
    parser.add_argument('input', help='输入音频文件路径')
    parser.add_argument('--output-dir', help='输出目录（可选）')
    parser.add_argument('--bit-rate', type=int, help='目标比特率（kbps，可选）')
    parser.add_argument('--preserve-quality', action='store_true', 
                       help='是否保留或提升原始质量')
    
    args = parser.parse_args()
    
    try:
        process_audio(
            args.input, 
            args.output_dir,
            args.bit_rate,
            args.preserve_quality
        )
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        exit(1)
    
    exit(0)

if __name__ == "__main__":
    main()
