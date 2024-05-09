# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import stable_whisper
import requests
import mimetypes

import json

filename = "input"


def cleanup_json_fields(file):
    root_fields_to_delete = ["regroup_history", "nonspeech_sections", "ori_dict"]
    segment_fields_to_keep = ["start", "end", "text"]
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for segment in list(data):
            if segment in root_fields_to_delete:
                del data[segment]

        for segment in data["segments"]:
            # trim segment string
            segment["text"] = segment["text"].strip()

            for field in list(segment):
                if field not in segment_fields_to_keep:
                    del segment[field]

    with open(file, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = stable_whisper.load_model("base")

    def predict(
        self,
        url: str = Input(description="Audio or video URL"),
        output_format: str = Input(
            choices=["ass", "json"],
            default="json",
            description="Output format: ass (ASS subtitles) or json (transcription in JSON format).",
        ),
    ) -> Path:
        """Download and save the file to disk"""
        response = requests.get(url)
        content_type = response.headers["Content-Type"]
        extension = mimetypes.guess_extension(content_type)
        with open(f"{filename}.{extension}", "wb") as outfile:
            outfile.write(response.content)

        """Run a single transcription on the model"""
        result = self.model.transcribe(f"{filename}.{extension}")

        print("✅ transcription complete")

        # TODO arg
        result.split_by_length(max_words=4)

        if output_format == "ass":
            result.to_ass(f"{filename}.ass")
            return Path(f"{filename}.ass")
        elif output_format == "json":
            result.save_as_json(f"{filename}.json")
            cleanup_json_fields(f"{filename}.json")
            return Path(f"{filename}.json")
