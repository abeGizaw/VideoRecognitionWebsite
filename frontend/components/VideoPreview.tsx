import { useRef, useEffect } from 'react';

interface VideoPreviewProps {
  stream: MediaStream | null;
  blob: string | undefined;
}

export const VideoPreview = ({ stream, blob }: VideoPreviewProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);
  if (!stream) {
    <video src={blob} controls autoPlay loop />;
  }
  return <video ref={videoRef} controls autoPlay loop />;
};
