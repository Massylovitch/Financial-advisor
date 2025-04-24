from bytewax.inputs import DynamicSource, StatelessSourcePartition
from streaming_pipeline import utils
from dotenv import load_dotenv
import datetime
import logging
import os
import requests

load_dotenv()
logger = logging.getLogger()


class AlpacaNewsBatchClient:

    NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

    def __init__(self, from_datetime, to_datetime) -> None:
        self._from_datetime = from_datetime
        self._to_datetime = to_datetime

        self._page_token = None

    def list(self):

        self._first_request = False

        # prepare the request URL
        headers = {
            "Apca-Api-Key-Id":  os.environ["ALPACA_API_KEY"],
            "Apca-Api-Secret-Key":  os.environ["ALPACA_API_SECRET"],
        }

        params = {
            "start": self._from_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": self._to_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 50,
            "include_content": True,
            "sort": "ASC",
        }
        if self._page_token is not None:
            params["page_token"] = self._page_token

        response = requests.get(self.NEWS_URL, headers=headers, params=params)

        # parse output
        next_page_token = None
        if response.status_code == 200:
            # parse response into json
            news_json = response.json()

            # extract next page token (if any)
            next_page_token = news_json.get("next_page_token", None)

        else:
            logger.error("Request failed with status code:", response.status_code)

        self._page_token = next_page_token

        return news_json["news"]

class AlpacaNewsBatchSource(StatelessSourcePartition):
    def __init__(self, from_datetime, to_datetime) -> None:
        
        self._alpaca_client = AlpacaNewsBatchClient(
            from_datetime=from_datetime, to_datetime=to_datetime
        )
        self.count = 0

    def next_batch(self):
        news = self._alpaca_client.list()
        if news is None or len(news) == 0 or self.count == 1:
            raise StopIteration()
        self.count += 1
        return [news]
    
class AlpacaNewsBatchInput(DynamicSource):
    def __init__(self, from_datetime, to_datetime) -> None:
        self._from_datetime = from_datetime
        self._to_datetime = to_datetime


    def build(self, step_id: str, worker_index: int, worker_count: int) -> StatelessSourcePartition:
        datetime_intervals = utils.split_time_range_into_intervals(
            from_datetime=self._from_datetime,
            to_datetime=self._to_datetime,
            n=worker_count,
        )
        worker_datetime_interval = datetime_intervals[worker_index]
        worker_from_datetime, worker_to_datetime = worker_datetime_interval

        return AlpacaNewsBatchSource(
                from_datetime=worker_from_datetime,
                to_datetime=worker_to_datetime,
            )



if __name__ == "__main__":

    to_datetime = datetime.datetime.now()
    from_datetime = to_datetime - datetime.timedelta(days=2)

    a = AlpacaNewsBatchInput(from_datetime, to_datetime)
    print(a.build())