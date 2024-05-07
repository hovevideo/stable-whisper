# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import stable_whisper
import requests
import mimetypes

# import json

filename = "input"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = stable_whisper.load_model("base")

    def predict(
        self,
        url: str = Input(description="File URL"),
        # scale: float = Input(
        #     description="Factor to scale image by", ge=0, le=10, default=1.5
        # ),
    ) -> Path:
        """Download and save the file to disk"""
        response = requests.get(url)
        content_type = response.headers["Content-Type"]
        extension = mimetypes.guess_extension(content_type)
        with open(f"{filename}.{extension}", "wb") as outfile:
            outfile.write(response.content)

        """Run a single transcription on the model"""
        result = self.model.transcribe(f"{filename}.{extension}")

        # TODO arg
        result.split_by_length(max_words=4)

        # TODO arg, choose output?
        result.to_ass(f"{filename}.ass")
        result.save_as_json(f"{filename}.json")
        return Path(f"{filename}.json")
        # with open(f"{filename}.json") as f:
        #     d = json.load(f)
        #     return d
