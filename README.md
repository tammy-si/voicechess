Python script that allows user to play chess on chess.com with voice command.

Audio processing done with PyAudio, webrtcvad, and OpenAI whisper.
Visual processing of the user's chessboard is done with OpenCV and a finetuned version of MobileNetV2
The audio and visual results are then feed into Google's Gemini AI API to determine users next move.
The move is then made with PyautoGUI.

Still a bit buggy. Perhaps that will change oneday.
