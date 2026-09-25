import asyncio
import edge_tts
from pydub import AudioSegment

dialogue_script = [
    {
        "speaker": "Doctor",
        "voice": "fa-IR-FaridNeural",
        "text": "سلام، روزتون بخیر. بفرمایید بنشینید، مشکلتون از چه زمانی شروع شده؟"
    },
    {
        "speaker": "Patient",
        "voice": "fa-IR-DilaraNeural",
        "text": "سلام آقای دکتر. از دیروز سوزش و درد شدید معده دارم و هر چی می‌خورم حالم بدتر می‌شه."
    },
    {
        "speaker": "Doctor",
        "voice": "fa-IR-FaridNeural",
        "text": "آیا سابقه مصرف داروی خاصی مثل آسپرین یا بروفن رو در این چند روز داشتید؟"
    },
    {
        "speaker": "Patient",
        "voice": "fa-IR-DilaraNeural",
        "text": "بله، برای دندون‌دردم چند تا ژلوفن مصرف کردم."
    }
]

async def build_audio():
    combined = AudioSegment.silent(duration=500)
    silence = AudioSegment.silent(duration=800)
    
    print("[+] Generating sample consultation audio...")
    for idx, turn in enumerate(dialogue_script):
        temp_file = f"temp_{idx}.mp3"
        comm = edge_tts.Communicate(turn["text"], turn["voice"])
        await comm.save(temp_file)
        
        audio = AudioSegment.from_file(temp_file)
        combined += audio + silence
    
    output_wav = "sample_consultation.wav"
    combined = combined.set_frame_rate(16000).set_channels(1)
    combined.export(output_wav, format="wav")
    print(f"[✓] Created '{output_wav}' successfully.")

if __name__ == "__main__":
    asyncio.run(build_audio())