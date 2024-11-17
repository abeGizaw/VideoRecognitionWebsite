import { Box, Sheet, Table, Typography } from "@mui/joy";
import { TextToSpeech } from "./TexToSpeech";
import { Message } from "../pages/whatTheVidDo/+Page";
import { ResultTable } from "./ResultTable";

export interface VideoInfoProps {
    displayMessage: Message;
    resultMessage: Message;
    textToSpeechBox: boolean;
}


export const VideoInfo = ({ displayMessage, textToSpeechBox, resultMessage }: VideoInfoProps) => {

    return (
        <Box sx={{ width: '100vh' }}>
            <Box>
                <Box display={textToSpeechBox ? 'block' : 'none'}>
                    <TextToSpeech text={resultMessage.message} type={resultMessage.type} />
                </Box>
                <Typography sx={{ textAlign: 'center' }}>{displayMessage.message}</Typography>
                <Box display={textToSpeechBox ? 'block' : 'none'}>
                    <ResultTable displayMessage={resultMessage.message} />
                </Box>
            </Box>

        </Box>
    )
}