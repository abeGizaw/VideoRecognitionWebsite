import { Box, Typography } from "@mui/joy";
import { TextToSpeech } from "./TexToSpeech";

export interface VideoInfoProps {
    message: string;
    textToSpeechBox: boolean;
}

export const VideoInfo = ({ message, textToSpeechBox }: VideoInfoProps) => {
    return (
        <>
            {message.length > 0 && <Box>
                <Box display={textToSpeechBox ? 'block' : 'none'}>
                    <TextToSpeech text={message} />
                </Box>
                <Typography>{message}</Typography>
            </Box>
            }
        </>
    )
}