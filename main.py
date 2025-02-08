# main.py
import os
import sys
import cv2 # type: ignore
import subprocess
import itertools
from config import COLOR_TONE_DICT, COLOR_TO_TONE_MAP, TONE_TO_SCALE, COLOR_MAP, BASIC_OCTAVE, LEFT_OCTAVE
from music_gen import generate_midi_from_basic_units, generate_lilypond_score_with_brace, process_basic_units, process_basic_units_new, group_notes_into_measures, generate_lilypond_single_staff, process_basic_units_from_top_colors, extract_main_melody_by_repetition, get_top_notes, create_intro_measures, create_empty_measures, create_intro_basic_measures, create_coda_basic_measures, extract_top_notes_and_durations, create_4x4_measures_from_top, create_coda_measures

def from_midi_to_mp3(image_name):
    # 构造生成的 MIDI 文件的路径
    midi_file = os.path.join(image_name)
    print("MIDI 文件生成：", midi_file)

    # 第二步：使用 FluidSynth 将 MIDI 文件转换为 WAV
    # 注意：这里使用 FluidR3_GM/FluidR3_GM.sf2 作为 SoundFont，请确保该文件存在且路径正确
    wav_file = os.path.join(image_name.replace(".midi", ".wav"))
    subprocess.run([
        "fluidsynth",
        "-ni", "FluidR3_GM/FluidR3_GM.sf2",  # 指定 SoundFont 文件
        midi_file,
        "-F", wav_file,
        "-r", "44100"
    ], check=True)
    print("WAV 文件生成：", wav_file)

    # 第三步：使用 ffmpeg 将 WAV 转换为 MP3
    mp3_file = os.path.join(image_name.replace(".midi", ".mp3"))
    subprocess.run([
        "ffmpeg",
        "-y",               # 如已有同名文件则覆盖
        "-i", wav_file,     # 输入 WAV 文件
        "-acodec", "libmp3lame",
        mp3_file
    ], check=True)
    print("MP3 文件生成：", mp3_file)


def main():
    if len(sys.argv) < 2:
        print("用法: python3 main.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"图像文件不存在: {image_path}")
        sys.exit(1)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像，请检查路径。")
    
    # 使用 os.path.basename 获取文件名，如 "IMG_8148.jpg"
    filename = os.path.basename(image_path)
    # 使用 os.path.splitext 分离扩展名，得到 "IMG_8148"
    image_name, _ = os.path.splitext(filename)
    

    # 读图 & resize
    image = cv2.imread(image_path)
    image_name = image_path.split("/")[1].split(".")[0]
    if image is None:
        raise ValueError("无法读取图像，请检查路径。")
    target_height = 500
    ratio = target_height / image.shape[0]
    new_width = int(image.shape[1]*ratio)
    image = cv2.resize(image,(new_width,target_height))

    # 1) 处理图像 -> basic_units
    basic_units = process_basic_units(image, COLOR_TO_TONE_MAP, COLOR_TONE_DICT) 
    # 2) 生成 MIDI
    os.makedirs(os.path.join("outputs", image_name), exist_ok=True)

    basic_midi_output = os.path.join("outputs", image_name , "basic.mid")
    basic_notes_list = generate_midi_from_basic_units(basic_units, basic_midi_output)

    flattened_basic_notes = list(itertools.chain(*basic_notes_list))
    # print(flattened_basic_notes)
    basic_notes = [(note, duration / 4.0, t, v) for (note, duration, t, v) in flattened_basic_notes]
    # print(basic_notes)
    grouped_basic_notes = group_notes_into_measures(basic_notes, beats_per_measure=4.0)
    # print(grouped_basic_notes)

    # output_basic_ly = os.path.join("outputs", image_name, "basic_single_staff.ly")
    # generate_lilypond_single_staff(grouped_basic_notes, BASIC_OCTAVE, output_basic_ly)
    # 调用 LilyPond
    # subprocess.run(["lilypond", output_basic_ly])

    

    # 3) 生成左手音符数
    left_units = process_basic_units_new(image, TONE_TO_SCALE, COLOR_MAP, grid_size=25)
    # 根据处理后的基本单元生成 MIDI 文件，并输出最终音符序列
    left_midi_output = os.path.join("outputs", image_name , "left.mid")
    left_notes_list = generate_midi_from_basic_units(left_units, left_midi_output)

    flattened_left_notes = list(itertools.chain(*left_notes_list))
    # print(flattened_left_notes)
    left_notes = [(note, duration / 4.0, t, v) for (note, duration, t, v) in flattened_left_notes]
    # print(left_notes)
    grouped_left_notes = group_notes_into_measures(left_notes, beats_per_measure=4.0)
    # print(grouped_left_notes)

    # output_left_ly = os.path.join("outputs", image_name, "left_single_staff.ly")
    # generate_lilypond_single_staff(grouped_left_notes, LEFT_OCTAVE, output_left_ly)
    # subprocess.run(["lilypond", output_left_ly])


    left_units_new = process_basic_units_from_top_colors(image, COLOR_MAP, TONE_TO_SCALE, num_colors=30, notes_per_color=16)
    left_midi_output_new = os.path.join("outputs", image_name , "left_new.mid")
    left_notes_new_list = generate_midi_from_basic_units(left_units_new, left_midi_output_new)

    flattened_left_notes_new = list(itertools.chain(*left_notes_new_list))
    # print(flattened_left_notes_new)
    left_notes_new = [(note, duration / 4.0, t, v) for (note, duration, t, v) in flattened_left_notes_new]
    # print(left_notes_new)
    grouped_left_notes_new = group_notes_into_measures(left_notes_new, beats_per_measure=4.0)
    # 如果 left 长于 right => 截断
    if len(grouped_left_notes_new) > len(grouped_basic_notes):
        grouped_left_notes_new = grouped_left_notes_new[:len(grouped_basic_notes)]
    # 如果 left 小于 right => 加空小节
    elif len(grouped_left_notes_new) < len(grouped_basic_notes):
        diff = len(grouped_basic_notes) - len(grouped_left_notes_new)
        extra_empty = create_empty_measures(diff)
        grouped_left_notes_new += extra_empty

    # output_left_new_ly = os.path.join("outputs", image_name, "left_new_single_staff.ly")
    # generate_lilypond_single_staff(grouped_left_notes_new, LEFT_OCTAVE, output_left_new_ly)
    # subprocess.run(["lilypond", output_left_new_ly])


    # 1) 抽取主体
    # 2) scale_for_intro : 10最重要音符
    scale_for_intro = get_top_notes(grouped_basic_notes, top_count=10)
    intro_end_vel = grouped_basic_notes[0][0][3]
    intro_tempo   = grouped_basic_notes[0][0][2]

    # 3) intro_right
    intro_basic = create_intro_measures(
        scale_notes=scale_for_intro,
        num_measures=4, 
        total_notes=16, 
        start_velocity=20, 
        end_velocity=intro_end_vel,
        ascending=True,
        tempo=intro_tempo
    )

    intro_basic = create_intro_basic_measures(
        scale_notes=scale_for_intro,
        num_measures=4,
        total_notes=16,
        start_velocity=10,               # 由更小(30)渐强到...
        end_velocity=intro_end_vel, 
        start_tempo=intro_tempo,
        ascending=True
    ) 

    top_notes, top_durs = extract_top_notes_and_durations(grouped_basic_notes, note_count=10, dur_count=5)
    intro_measures = create_4x4_measures_from_top(
        note_list=top_notes,
        dur_list=top_durs,
        seed=42,
        total_measures=4,
        tempo=intro_tempo,
        velocity=intro_end_vel  
    )

    # 4) coda_right
    coda_start_vel = grouped_basic_notes[-1][-1][3]
    coda_tempo     = grouped_basic_notes[-1][-1][2]

    coda_basic = create_coda_basic_measures(
        scale_notes=scale_for_intro,
        num_measures=4,
        total_notes=16,
        start_velocity=coda_start_vel,
        end_velocity=10,                # 渐弱到30
        start_tempo=coda_tempo,
        descending=True
    )

    top_notes, top_durs = extract_top_notes_and_durations(grouped_basic_notes, note_count=10, dur_count=5)
    main_last_note = grouped_basic_notes[-1][-1]
    coda_measures = create_coda_measures(
        note_list=top_notes,
        dur_list=top_durs,
        seed=999,
        total_measures=4,
        base_tempo=coda_tempo,
        base_velocity=coda_start_vel,
        main_last_note=main_last_note
    )

    # 4) 合并

    # full_right = intro_measures + grouped_basic_notes + coda_measures

    num_intro_measures = len(intro_basic)
    num_coda_measures  = len(coda_basic)

    intro_left  = create_empty_measures(num_intro_measures, tempo=80, velocity=0)
    coda_left   = create_empty_measures(num_coda_measures,  tempo=80, velocity=0)

    full_left = intro_left + grouped_left_notes + coda_left
    full_left_new = intro_left + grouped_left_notes_new + coda_left
    
    print(len(grouped_basic_notes), len(grouped_left_notes), len(grouped_left_notes_new))
    

    # 4) 生成 LilyPond
    ly_filename = os.path.join("outputs", image_name, os.path.splitext(os.path.basename(image_path))[0] + "-score.ly")
    # generate_lilypond_score_with_brace(grouped_basic_notes, BASIC_OCTAVE, grouped_left_notes, LEFT_OCTAVE, filename=ly_filename)
    # generate_lilypond_score_with_brace(full_right, BASIC_OCTAVE, full_left, LEFT_OCTAVE, filename=ly_filename)
    generate_lilypond_score_with_brace(intro_measures, grouped_basic_notes, coda_measures, BASIC_OCTAVE, full_left, LEFT_OCTAVE, filename=ly_filename)
    # generate_lilypond_with_intro_coda(intro_measures, grouped_basic_notes, coda_measures, BASIC_OCTAVE, full_left, LEFT_OCTAVE, filename=ly_filename)
                        

    # 5) 调用 LilyPond 和 ImageMagick
    subprocess.run(["lilypond", "-o", os.path.join("outputs", image_name), ly_filename])
    from_midi_to_mp3(ly_filename.replace(".ly", ".midi"))
    
    pdf_file = ly_filename.replace(".ly", ".pdf")
    png_file = ly_filename.replace(".ly", ".png")
    subprocess.run(["magick", "-density", "300", pdf_file, png_file])


    ly_filename_new = os.path.join("outputs", image_name, os.path.splitext(os.path.basename(image_path))[0] + "-score-new.ly")
    # generate_lilypond_score_with_brace(grouped_basic_notes, BASIC_OCTAVE, grouped_left_notes_new, LEFT_OCTAVE, filename=ly_filename_new)
    # generate_lilypond_score_with_brace(full_right, BASIC_OCTAVE, full_left_new, LEFT_OCTAVE, filename=ly_filename_new)
    generate_lilypond_score_with_brace(intro_measures, grouped_basic_notes, coda_measures, BASIC_OCTAVE, full_left_new, LEFT_OCTAVE, filename=ly_filename_new)
    # generate_lilypond_with_intro_coda(intro_measures, grouped_basic_notes, coda_measures, BASIC_OCTAVE, full_left_new, LEFT_OCTAVE, filename=ly_filename)


    subprocess.run(["lilypond", "-o", os.path.join("outputs", image_name), ly_filename_new])
    from_midi_to_mp3(ly_filename_new.replace(".ly", ".midi"))

    pdf_file_new = ly_filename_new.replace(".ly", ".pdf")
    png_file_new = ly_filename_new.replace(".ly", ".png")
    subprocess.run(["magick", "-density", "300", pdf_file_new, png_file_new])


if __name__=="__main__":
    main()