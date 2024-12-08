import os
from gtts import gTTS
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor


# Function to convert text to speech and save as MP3
def text_to_speech(text, output_file, accent):
    tts = gTTS(text=text, lang="en", tld=accent, slow=False)
    tts.save(output_file)


# Function to adjust the speed of an audio segment
def adjust_speed(audio, speed_change):
    # Speed up the audio by the specified factor
    return audio._spawn(
        audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * speed_change)}
    )


# Function to combine HOST and GUEST audio
def combine_audio(host_audio_files, guest_audio_files, output_file):
    final_audio = AudioSegment.empty()
    pause = AudioSegment.silent(duration=300)  # 0.3s pause between lines

    ## Add intro music
    intro_music = AudioSegment.from_mp3("./assests/intro-1.mp3")

    final_audio += intro_music
    final_audio += pause  # Add pause after intro music

    host_index = 0
    guest_index = 0

    while host_index < len(host_audio_files) or guest_index < len(guest_audio_files):
        if host_index < len(host_audio_files):
            host_audio = AudioSegment.from_mp3(host_audio_files[host_index])
            host_audio = adjust_speed(host_audio, 1.2)  # Increase speed by 20%
            final_audio += host_audio
            host_index += 1
            final_audio += pause  # Add pause between lines

        if guest_index < len(guest_audio_files):
            guest_audio = AudioSegment.from_mp3(guest_audio_files[guest_index])
            guest_audio = adjust_speed(guest_audio, 1.2)  # Increase speed by 20%
            final_audio += guest_audio
            guest_index += 1
            final_audio += pause  # Add pause between lines

    # Export the final audio file as MP3
    final_audio.export(output_file, format="mp3")

    # Clean up temporary files
    for file in host_audio_files + guest_audio_files:
        os.remove(file)

    print(f"Final audio file generated: {output_file}")


# Function to process and combine audio
def process_audio(
    script, host_accent, guest_accent, output_file_path="final_podcast_output.mp3"
):
    # Split the script into HOST and GUEST parts
    lines = script.split("\n")
    host_lines = []
    guest_lines = []

    current_speaker = None

    for line in lines:
        if line.startswith("HOST:"):
            current_speaker = "HOST"
            host_lines.append(line[6:])
        elif line.startswith("GUEST:"):
            current_speaker = "GUEST"
            guest_lines.append(line[7:])
        else:
            if current_speaker == "HOST":
                host_lines.append(line)
            elif current_speaker == "GUEST":
                guest_lines.append(line)

    # Convert HOST lines to speech using multiprocessing
    host_audio_files = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i, line in enumerate(host_lines):
            if line.strip():  # Ensure the line is not empty
                output_file = f"host_output_{i}.mp3"
                host_audio_files.append(output_file)
                futures.append(
                    executor.submit(text_to_speech, line, output_file, host_accent)
                )
        for future in futures:
            future.result()

    # Convert GUEST lines to speech using multiprocessing
    guest_audio_files = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, line in enumerate(guest_lines):
            if line.strip():  # Ensure the line is not empty
                output_file = f"guest_output_{i}.mp3"
                guest_audio_files.append(output_file)
                futures.append(
                    executor.submit(text_to_speech, line, output_file, guest_accent)
                )
        for future in futures:
            future.result()

    print("Audio files generated for HOST and GUEST")

    # Combine HOST and GUEST audio
    combine_audio(host_audio_files, guest_audio_files, output_file_path)

    return output_file_path
