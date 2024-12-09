import React, { useState, useEffect } from "react";
import { Box } from "@mui/joy";

export interface TextToSpeechProps {
    text: string;
}

export const TextToSpeech = ({ text }: TextToSpeechProps) => {
    const [isPaused, setIsPaused] = useState(false);
    const [utterance, setUtterance] = useState<SpeechSynthesisUtterance>();
    const [voice, setVoice] = useState<SpeechSynthesisVoice>();
    const [pitch, setPitch] = useState(1);
    const [rate, setRate] = useState(1);
    const [volume, setVolume] = useState(.2);

    useEffect(() => {
        const synth = window.speechSynthesis;
        const u = new SpeechSynthesisUtterance(text);
        const voices: SpeechSynthesisVoice[] = synth.getVoices();
        setUtterance(u);
        setVoice(voices[0]);


        return () => {
           synth.cancel();
        };
    }, [text]);

    const handlePlay = () => {
        const synth = window.speechSynthesis;


        if (isPaused) {
            synth.resume();
        } else if (utterance) {
            utterance.voice = voice || null;
            utterance.pitch = pitch;
            utterance.rate = rate;
            utterance.volume = volume;
            synth.speak(utterance);
        }

        setIsPaused(false);
    };

    const handlePause = () => {
        const synth = window.speechSynthesis;

        synth.pause();

        setIsPaused(true);
    };

    const handleStop = () => {
        const synth = window.speechSynthesis;

        synth.cancel();

        setIsPaused(false);
    };

    const handleVoiceChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const voices = window.speechSynthesis.getVoices();
        setVoice(voices.find((v) => v.name === event.target.value));
    };

    const handlePitchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setPitch(parseFloat(event.target.value));
    };

    const handleRateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setRate(parseFloat(event.target.value));
    };

    const handleVolumeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setVolume(parseFloat(event.target.value));
    };

    return (
        <Box >
            <label>
                Voice:
                <select value={voice?.name} onChange={handleVoiceChange}>
                    {window.speechSynthesis.getVoices().map((voice) => (
                        <option key={voice.name} value={voice.name}>
                            {voice.name}
                        </option>
                    ))}
                </select>
            </label>

            <br />

            <label>
                Pitch:
                <input
                    type="range"
                    min="0.5"
                    max="2"
                    step="0.1"
                    value={pitch}
                    onChange={handlePitchChange}
                />
            </label>

            <br />

            <label>
                Speed:
                <input
                    type="range"
                    min="0.5"
                    max="2"
                    step="0.1"
                    value={rate}
                    onChange={handleRateChange}
                />
            </label>
            <br />
            <label>
                Volume:
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={volume}
                    onChange={handleVolumeChange}
                />
            </label>

            <br />

            <button onClick={handlePlay}>{isPaused ? "Resume" : "Play"}</button>
            <button onClick={handlePause}>Pause</button>
            <button onClick={handleStop}>Stop</button>
        </Box>
    );
};

