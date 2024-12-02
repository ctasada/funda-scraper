"""Main funda scraper module"""

import argparse
import datetime
import json
import multiprocessing as mp
import os
from collections import OrderedDict
import time
from typing import List, Optional
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from funda_scraper.config.core import config
from funda_scraper.preprocess import clean_date_format, preprocess_data
from funda_scraper.utils import logger


class FundaScraper(object):
    """
    A class used to scrape real estate data from the Funda website.
    """

    def __init__(
        self,
        area: str,
        want_to: str,
        page_start: int = 1,
        n_pages: int = 1,
        find_past: bool = False,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        days_since: Optional[int] = None,
        property_type: Optional[str] = None,
        min_floor_area: Optional[str] = None,
        max_floor_area: Optional[str] = None,
        min_rooms: Optional[str] = None,
        max_rooms: Optional[str] = None,
        construction_period: Optional[str] = None,
        sort: Optional[str] = None,
    ):
        """
        :param area: The area to search for properties, this can be a comma-seperated list, formatted for URL compatibility.
        :param want_to: Specifies whether the user wants to buy or rent properties.
        :param page_start: The starting page number for the search.
        :param n_pages: The number of pages to scrape.
        :param find_past: Flag to indicate whether to find past listings.
        :param min_price: The minimum price for the property search.
        :param max_price: The maximum price for the property search.
        :param days_since: The maximum number of days since the listing was published.
        :param property_type: The type of property to search for.
        :param min_floor_area: The minimum floor area for the property search.
        :param max_floor_area: The maximum floor area for the property search.
        :param min_rooms: The minimum number of rooms for the property search.
        :param max_rooms: The maximum number of rooms for the property search.
        :param construction_period: The construction period for the property search.
        :param sort: The sorting criterion for the search results.
        """
        # Init attributes
        self.area = area.lower().replace(" ", "-").replace(",","\",\"") #added functionality to add multiple cities, seperated by ', '
        self.property_type = property_type
        self.want_to = want_to
        self.find_past = find_past
        self.page_start = max(page_start, 1)
        self.n_pages = max(n_pages, 1)
        self.page_end = self.page_start + self.n_pages - 1
        self.min_price = min_price
        self.max_price = max_price
        self.days_since = days_since
        self.min_floor_area = min_floor_area
        self.max_floor_area = max_floor_area
        self.min_rooms = min_rooms
        self.max_rooms = max_rooms
        self.construction_period = construction_period
        self.sort = sort

        # Instantiate along the way
        self.links: List[str] = []
        self.raw_df = pd.DataFrame()
        self.clean_df = pd.DataFrame()
        self.base_url = config.base_url
        self.selectors = config.css_selector

    def __repr__(self):
        return (
            f"FundaScraper(area={self.area}, "
            f"want_to={self.want_to}, "
            f"n_pages={self.n_pages}, "
            f"page_start={self.page_start}, "
            f"find_past={self.find_past}, "
            f"min_price={self.min_price}, "
            f"max_price={self.max_price}, "
            f"days_since={self.days_since}, "
            f"min_floor_area={self.min_floor_area}, "
            f"max_floor_area={self.max_floor_area}, "
            f"min_rooms={self.min_rooms}, "
            f"max_rooms={self.max_rooms}, "
            f"find_past={self.find_past})"
            f"min_price={self.min_price})"
            f"max_price={self.max_price})"
            f"days_since={self.days_since})"
            f"sort={self.sort})"
        )

    @property
    def to_buy(self) -> bool:
        """Determines if the search is for buying or renting properties."""
        if self.want_to.lower() in ["buy", "koop", "b", "k"]:
            return True
        elif self.want_to.lower() in ["rent", "huur", "r", "h"]:
            return False
        else:
            raise ValueError("'want_to' must be either 'buy' or 'rent'.")

    @property
    def check_days_since(self) -> int:
        """Validates the 'days_since' attribute."""
        if self.find_past:
            raise ValueError("'days_since' can only be specified when find_past=False.")

        if self.days_since in [None, 1, 3, 5, 10, 30]:
            return self.days_since
        else:
            raise ValueError("'days_since' must be either None, 1, 3, 5, 10 or 30.")

    @property
    def check_sort(self) -> str:
        """Validates the 'sort' attribute."""
        if self.sort in [
            None,
            "relevancy",
            "date_down",
            "date_up",
            "price_up",
            "price_down",
            "floor_area_down",
            "plot_area_down",
            "city_up",
            "postal_code_up",
        ]:
            return self.sort
        else:
            raise ValueError(
                "'sort' must be either None, 'relevancy', 'date_down', 'date_up', 'price_up', 'price_down', "
                "'floor_area_down', 'plot_area_down', 'city_up' or 'postal_code_up'. "
            )

    @staticmethod
    def _check_dir() -> None:
        """Ensures the existence of the directory for storing data."""
        if not os.path.exists("data"):
            os.makedirs("data")

    @staticmethod
    def _get_links_from_one_parent(url: str) -> List[str]:
        """Scrapes all available property links from a single Funda search page."""
        response = requests.get(url, headers=config.header)
        soup = BeautifulSoup(response.text, "lxml")

        script_tag = soup.find_all("script", {"type": "application/ld+json"})[0]
        json_data = json.loads(script_tag.contents[0])
        urls = [item["url"] for item in json_data["itemListElement"]]
        return urls

    def reset(
        self,
        area: Optional[str] = None,
        property_type: Optional[str] = None,
        want_to: Optional[str] = None,
        page_start: Optional[int] = None,
        n_pages: Optional[int] = None,
        find_past: Optional[bool] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        days_since: Optional[int] = None,
        min_floor_area: Optional[str] = None,
        max_floor_area: Optional[str] = None,
        min_rooms: Optional[str] = None,
        max_rooms: Optional[str] = None,
        construction_period: Optional[str] = None,
        sort: Optional[str] = None,
    ) -> None:
        """Resets or initializes the search parameters."""
        if area is not None:
            self.area = area
        if property_type is not None:
            self.property_type = property_type
        if want_to is not None:
            self.want_to = want_to
        if page_start is not None:
            self.page_start = max(page_start, 1)
        if n_pages is not None:
            self.n_pages = max(n_pages, 1)
        if find_past is not None:
            self.find_past = find_past
        if min_price is not None:
            self.min_price = min_price
        if max_price is not None:
            self.max_price = max_price
        if days_since is not None:
            self.days_since = days_since
        if min_floor_area is not None:
            self.min_floor_area = min_floor_area
        if max_floor_area is not None:
            self.max_floor_area = max_floor_area
        if min_rooms is not None:
            self.min_rooms = min_rooms
        if max_rooms is not None:
            self.max_rooms = max_rooms
        if construction_period is not None:
            self.construction_period = construction_period
        if sort is not None:
            self.sort = sort

    @staticmethod
    def remove_duplicates(lst: List[str]) -> List[str]:
        """Removes duplicate links from a list."""
        return list(OrderedDict.fromkeys(lst))

    @staticmethod
    def fix_link(link: str) -> str:
        if not link.startswith("https://www.funda.nl/en"):
            link = link.replace("https://www.funda.nl/", "https://www.funda.nl/en/")

        return link

    def fetch_all_links(self, page_start: int = None, n_pages: int = None) -> None:
        """Collects all available property links across multiple pages."""

        page_start = self.page_start if page_start is None else page_start
        n_pages = self.n_pages if n_pages is None else n_pages

        logger.info("*** Phase 1: Fetch all the available links from all pages *** ")
        urls = []
        main_url = self._build_main_query_url()

        for i in tqdm(range(page_start, page_start + n_pages)):
            try:
                item_list = self._get_links_from_one_parent(
                    f"{main_url}&search_result={i}"
                )
                urls += item_list
                time.sleep(.2) # short sleep to reduce the chance of getting locked out of Funda
            except IndexError:
                self.page_end = i
                logger.info(f"*** The last available page is {self.page_end} ***")
                break

        urls = self.remove_duplicates(urls)
        fixed_urls = [self.fix_link(url) for url in urls]

        logger.info(
            f"*** Got all the urls. {len(fixed_urls)} houses found from {self.page_start} to {self.page_end} ***"
        )
        self.links = fixed_urls

    def _build_main_query_url(self) -> str:
        """Constructs the main query URL for the search."""
        query = "koop" if self.to_buy else "huur"

        main_url = (
            f"{self.base_url}/zoeken/{query}?"
        )

        areas = self.area.split(",")
        formatted_areas = [
            "%22" + area + "%22" for area in areas
        ]
        main_url += f"selected_area=%5B{','.join(formatted_areas)}%5D"

        if self.property_type:
            property_types = self.property_type.split(",")
            formatted_property_types = [
                "%22" + prop_type + "%22" for prop_type in property_types
            ]
            main_url += f"&object_type=%5B{','.join(formatted_property_types)}%5D"

        main_url = f'{main_url}&availability=%5B"available"%5D'

        if self.min_price is not None or self.max_price is not None:
            min_price = "" if self.min_price is None else self.min_price
            max_price = "" if self.max_price is None else self.max_price
            main_url = f"{main_url}&price=%22{min_price}-{max_price}%22"

        if self.days_since is not None:
            main_url = f"{main_url}&publication_date={self.check_days_since}"

        if self.min_floor_area or self.max_floor_area:
            min_floor_area = "" if self.min_floor_area is None else self.min_floor_area
            max_floor_area = "" if self.max_floor_area is None else self.max_floor_area
            main_url = f"{main_url}&floor_area=%22{min_floor_area}-{max_floor_area}%22"

        if self.min_rooms or self.max_rooms:
            min_rooms = "" if self.min_rooms is None else self.min_rooms
            max_rooms = "" if self.max_rooms is None else self.max_rooms
            main_url = f"{main_url}&rooms=%22{min_rooms}-{max_rooms}%22"

        if self.sort is not None:
            main_url = f"{main_url}&sort=%22{self.check_sort}%22"

        if self.construction_period:
            construction_periods = self.construction_period.split(",")
            formatted_construction_periods = [
                "%22" + period + "%22" for period in construction_periods
            ]
            main_url += f"&construction_period=%5B{','.join(formatted_construction_periods)}%5D"

        logger.info(f"*** Main URL: {main_url} ***")
        return main_url

    @staticmethod
    def get_value_from_data(soup: BeautifulSoup, selector: str) -> str:
        """Extracts data from HTML using the data structure."""
        if selector == "":
            return "na"

        if type(selector) != list:
            selector = [selector]

        for s in selector:
            try:
                result_element = soup.find("dt", string=s).find_next("dd")
                result = result_element.text.strip() if result_element else "na"
                if result != "na":
                    break
            except AttributeError:
                result = "na"

        return result

    @staticmethod
    def get_address_value(soup: BeautifulSoup) -> dict:
        """Extracts the address value from the HTML."""
        div_element = soup.find('div', attrs={
            'neighborhoodidentifier': True,
        })

        # Extract street address (first span)
        street_address = div_element.find('span')

        return {
            "neighborhood_name": div_element.get('neighborhoodidentifier', 'na'),
            "city": div_element.get('city', 'na'),
            "zip_code": div_element.get('postcode', 'na'),
            "housenumber": div_element.get('housenumber', 'na'),
            "province": div_element.get('province', 'na'),
            "address": street_address.text if street_address else "na",
        }

    def scrape_one_link(self, link: str) -> List[str]:
        """Scrapes data from a single property link."""

        # Initialize for each page
        response = requests.get(link, headers=config.header)
        soup = BeautifulSoup(response.text, "lxml")

        result = [
            link,
            self.get_value_from_data(soup, self.selectors.price).replace("kosten koper", "").strip(),
            self.get_address_value(soup)['address'],
            self.get_address_value(soup)['zip_code'],
            self.get_value_from_data(soup, self.selectors.size).replace("m²", "").strip(),
            self.get_value_from_data(soup, self.selectors.year),
            self.get_value_from_data(soup, self.selectors.living_area).replace("m²", "").strip(),
            self.get_value_from_data(soup, self.selectors.kind_of_house),
            self.get_value_from_data(soup, self.selectors.building_type),
            self.get_value_from_data(soup, self.selectors.num_of_rooms),
            self.get_value_from_data(soup, self.selectors.num_of_bathrooms),
            self.get_value_from_data(soup, self.selectors.energy_label).replace("What does this mean?", "").strip(),
            self.get_value_from_data(soup, self.selectors.insulation),
            self.get_value_from_data(soup, self.selectors.heating),
            self.get_value_from_data(soup, self.selectors.ownership),
            self.get_value_from_data(soup, self.selectors.parking),
            self.get_address_value(soup)['city'],
            self.get_address_value(soup)['neighborhood_name'],
        ]

        return result

    def scrape_pages(self) -> None:
        """Scrapes data from all collected property links."""

        logger.info("*** Phase 2: Start scraping from individual links ***")
        df = pd.DataFrame({key: [] for key in self.selectors.keys()})

        # Scrape pages with multiprocessing to improve efficiency
        # TODO: use asyncio instead
        pools = mp.cpu_count()
        content = process_map(self.scrape_one_link, self.links, max_workers=pools)

        for i, c in enumerate(content):
            df.loc[len(df)] = c

        logger.info(f"*** All scraping done: {df.shape[0]} results ***")
        self.raw_df = df

    def save_csv(self, df: pd.DataFrame, filepath: str = None) -> None:
        """Saves the scraped data to a CSV file."""
        if filepath is None:
            self._check_dir()
            date = str(datetime.datetime.now().date()).replace("-", "")
            want_to = "buy" if self.to_buy else "rent"
            filepath = f"./data/houseprice_{date}_{self.area}_{want_to}_{len(self.links)}.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"*** File saved: {filepath}. ***")

    def run(
        self, raw_data: bool = False, save: bool = False, filepath: str = None
    ) -> pd.DataFrame:
        """
        Runs the full scraping process, optionally saving the results to a CSV file.

        :param raw_data: if true, the data won't be pre-processed
        :param save: if true, the data will be saved as a csv file
        :param filepath: the name for the file
        :return: the (pre-processed) dataframe from scraping
        """
        self.fetch_all_links()
        self.scrape_pages()

        if raw_data:
            df = self.raw_df
        else:
            logger.info("*** Cleaning data ***")
            df = preprocess_data(df=self.raw_df, is_past=self.find_past)
            self.clean_df = df

        if save:
            self.save_csv(df, filepath)

        logger.info("*** Done! ***")
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--area",
        type=str,
        help="Specify which area you are looking for",
        default="amsterdam",
    )
    parser.add_argument(
        "--want_to",
        type=str,
        help="Specify you want to 'rent' or 'buy'",
        default="rent",
        choices=["rent", "buy"],
    )
    parser.add_argument(
        "--find_past",
        action="store_true",
        help="Indicate whether you want to use historical data",
    )
    parser.add_argument(
        "--page_start", type=int, help="Specify which page to start scraping", default=1
    )
    parser.add_argument(
        "--n_pages", type=int, help="Specify how many pages to scrape", default=1
    )
    parser.add_argument(
        "--min_price", type=int, help="Specify the min price", default=None
    )
    parser.add_argument(
        "--max_price", type=int, help="Specify the max price", default=None
    )
    parser.add_argument(
        "--min_floor_area", type=int, help="Indicate the minimum floor area", default=None
    )
    parser.add_argument(
        "--max_floor_area", type=int, help="Indicate the maximum floor area", default=None
    )
    parser.add_argument(
        "--min_rooms", type=int, help="Indicate the minimum number of rooms", default=None
    )
    parser.add_argument(
        "--max_rooms", type=int, help="Indicate the maximum number of rooms", default=None
    )
    parser.add_argument(
        "--construction_period", type=str, help="Specify the desired construction periods", default=None
    )
    parser.add_argument(
        "--property_type", type=str, help="Specify the desired property type(s)", default=None
    )
    parser.add_argument(
        "--days_since",
        type=int,
        help="Specify the days since publication",
        default=None,
    )
    parser.add_argument(
        "--sort",
        type=str,
        help="Specify sorting",
        default=None,
        choices=[
            None,
            "relevancy",
            "date_down",
            "date_up",
            "price_up",
            "price_down",
            "floor_area_down",
            "plot_area_down",
            "city_up" "postal_code_up",
        ],
    )
    parser.add_argument(
        "--raw_data",
        action="store_true",
        help="Indicate whether you want the raw scraping result",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Indicate whether you want to save the data",
    )

    args = parser.parse_args()
    scraper = FundaScraper(
        area=args.area,
        want_to=args.want_to,
        find_past=args.find_past,
        page_start=args.page_start,
        n_pages=args.n_pages,
        min_price=args.min_price,
        max_price=args.max_price,
        days_since=args.days_since,
        min_floor_area=args.min_floor_area,
        max_floor_area=args.max_floor_area,
        min_rooms=args.min_rooms,
        max_rooms=args.max_rooms,
        construction_period=args.construction_period,
        property_type=args.property_type,
        sort=args.sort,
    )
    df = scraper.run(raw_data=args.raw_data, save=args.save)
    print(df.head())
