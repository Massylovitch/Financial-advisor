import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from dotenv import load_dotenv

import requests
from utils.logger import get_logger
from utils.paths import DATA_DIR

logger = get_logger()
load_dotenv()


try:
    ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
    ALPACA_API_SECRET = os.environ["ALPACA_API_SECRET"]
except KeyError:
    logger.error(
        "Please set the environment variables ALPACA_API_KEY and ALPACA_API_SECRET"
    )
    raise


@dataclass
class News:
    headline: str
    summary: str
    content: str
    date: datetime


def fetch_batch_of_news(
    from_date,
    to_date,
    page_token= None,
):

    headers = {
        "Apca-Api-Key-Id": ALPACA_API_KEY,
        "Apca-Api-Secret-Key": ALPACA_API_SECRET,
    }
    params = {
        "start": from_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": to_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 50,
        "include_content": True,
        "sort": "ASC",
    }
    if page_token is not None:
        params["page_token"] = page_token

    url = "https://data.alpaca.markets/v1beta1/news"

    response = requests.get(url, headers=headers, params=params)

    list_of_news = []
    next_page_token = None
    if response.status_code == 200:  # Check if the request was successful
        news_json = response.json()

        next_page_token = news_json.get("next_page_token", None)

        for n in news_json["news"]:
            list_of_news.append(
                News(
                    headline=n["headline"],
                    date=n["updated_at"],
                    summary=n["summary"],
                    content=n["content"],
                )
            )

    else:
        logger.error("Request failed with status code:", response.status_code)

    return list_of_news, next_page_token


def save_news_to_json(news_list, filename):
    news_data = [
        {
            "headline": news.headline,
            "date": news.date,
            "summary": news.summary,
            "content": news.content,
        }
        for news in news_list
    ]
    with open(filename, "w") as json_file:
        json.dump(news_data, json_file, indent=4)


def download_historical_news(
    from_date,
    to_date,
):
    logger.info(f"Downloading historical news from {from_date} to {to_date}")
    list_of_news, next_page_token = fetch_batch_of_news(from_date, to_date)
    logger.info(f"Fetched {len(list_of_news)} news")
    logger.debug(f"Next page token: {next_page_token}")

    while next_page_token is not None:
        batch_of_news, next_page_token = fetch_batch_of_news(
            from_date, to_date, next_page_token
        )
        list_of_news += batch_of_news
        logger.info(f"Fetched a total of {len(list_of_news)} news")
        logger.debug(f"Next page token: {next_page_token}")

        logger.debug(f"Last date in batch: {batch_of_news[-1].date}")

    path_to_file = (
        DATA_DIR
        / f'news_{from_date.strftime("%Y-%m-%d")}_{to_date.strftime("%Y-%m-%d")}.json'
    )
    save_news_to_json(list_of_news, path_to_file)

    logger.info(f"News data saved to {path_to_file}")

    return path_to_file


if "__main__" == __name__:
    from_date = datetime(2023, 1, 1)
    to_date = datetime(2023, 1, 5)

    download_historical_news(from_date, to_date)
