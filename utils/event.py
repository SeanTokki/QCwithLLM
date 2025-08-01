import json
import boto3
import os
from dotenv import load_dotenv


class EventBridge:
    BUS_NAME = "default"

    def __init__(self):
        load_dotenv(".env")
        
        self.events = boto3.client(
            "events",
            region_name="ap-northeast-2",
            aws_access_key_id=os.getenv("AWS_EVENT_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_EVENT_SECRET_ACCESS_KEY"),
        )

    def send(self, task_name: str, params: dict):
        return self.events.put_events(
            Entries=[
                {
                    "Source": "datepop.crawler",
                    "DetailType": "RunPython",
                    "Detail": json.dumps({"task": task_name, "params": params}),
                    "EventBusName": self.BUS_NAME,
                }
            ]
        )