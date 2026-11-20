from pydantic import BaseModel, Field
from typing import List, Dict, Self
from bs4 import BeautifulSoup
import re
import feedparser
from tqdm import tqdm
import requests
import time

feeds = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
    "https://vipoutlet.com/collections/category-electronics.atom",
    "https://vipoutlet.com/collections/category-electronics-computers-laptops.atom"
    
]

# You could also add: "https://www.dealnews.com/c238/Automotive/?rss=1"
# "https://www.dealnews.com/c196/Home-Garden/?rss=1"


def extract(html_snippet: str) -> str:
    """
    Use Beautiful Soup to clean up this HTML snippet and extract useful text
    """
    soup = BeautifulSoup(html_snippet, "html.parser")
    snippet_div = soup.find("div", class_="snippet summary")

    if snippet_div:
        description = snippet_div.get_text(strip=True)
        description = BeautifulSoup(description, "html.parser").get_text()
        description = re.sub("<[^<]+?>", "", description)
        result = description.strip()
    else:
        result = html_snippet
    return result.replace("\n", " ")


class ScrapedDeal:
    """
    A class to represent a Deal retrieved from an RSS feed
    """

    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str
    available: bool

    def __init__(self, entry: Dict[str, str]):
        """
        Populate this instance based on the provided dict
        """
        self.title = entry["title"]
        self.summary = extract(entry.get("summary", ""))
        self.url = entry["links"][0]["href"]
        self.available = True

        if "vipoutlet.com/products/" in self.url:
            self._populate_shopify()
        else:
            self._populate_html()

        self.truncate()

    def _populate_shopify(self):
        """
        Fetch live variant pricing, in-stock status, and specs from Shopify's .js endpoint
        """
        resp = requests.get(f"{self.url.rstrip('/')}.js", timeout=15)
        if resp.ok:
            data = resp.json()
            in_stock = [v for v in data.get("variants", []) if v.get("available")]
            if not in_stock:
                self.available = False
                self.details = ""
                self.features = ""
                return
            best_variant = min(in_stock, key=lambda v: v["price"])
            price_dollars = best_variant["price"] / 100.0
            condition = best_variant.get("title", "")
            raw_desc = BeautifulSoup(data.get("description", ""), "html.parser").get_text(" ", strip=True)
            content = f"Price: ${price_dollars:.2f} (Condition: {condition}). {raw_desc}"
        else:
            content = BeautifulSoup(self.summary, "html.parser").get_text(" ", strip=True)

        if "Product Details:" in content:
            self.details, self.features = content.split("Product Details:", 1)
        elif "Features" in content:
            self.details, self.features = content.split("Features", 1)
        else:
            self.details = content
            self.features = ""

    def _populate_html(self):
        """
        Fetch DealNews (or generic RSS) HTML page and extract details + features
        """
        stuff = requests.get(self.url, timeout=15).content
        soup = BeautifulSoup(stuff, "html.parser")
        section = soup.find("div", class_="content-section")
        if section:
            content = section.get_text()
        else:
            content = BeautifulSoup(self.summary, "html.parser").get_text(" ", strip=True)
        content = content.replace("\nmore", "").replace("\n", " ")
        if "Features" in content:
            self.details, self.features = content.split("Features", 1)
        else:
            self.details = content
            self.features = ""

    def truncate(self):
        """
        Limit the fields to a sensible length to avoid sending too much info to the model
        """
        self.title = self.title[:100]
        self.details = self.details[:500]
        self.features = self.features[:500]

    def __repr__(self):
        """
        Return a string to describe this deal
        """
        return f"<{self.title}>"

    def describe(self):
        """
        Return a longer string to describe this deal for use in calling a model
        """
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

    @classmethod
    def fetch(cls, show_progress: bool = False) -> List[Self]:
        """
        Retrieve all deals from the selected RSS feeds
        """
        deals = []
        seen_urls = set()
        feed_iter = tqdm(feeds) if show_progress else feeds
        for feed_url in feed_iter:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                url = entry["links"][0]["href"]
                if url in seen_urls:
                    continue
                deal = cls(entry)
                if deal.available:
                    deals.append(deal)
                    seen_urls.add(url)
                time.sleep(0.05)
        return deals


class Deal(BaseModel):
    """
    A class to Represent a Deal with a summary description
    """

    product_description: str = Field(
        description="Your clearly expressed summary of the product in 3-4 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a short paragraph of text for each item you choose."
    )
    price: float = Field(
        description="The actual price of this product, as advertised in the deal. Be sure to give the actual price; for example, if a deal is described as $100 off the usual $300 price, you should respond with $200"
    )
    url: str = Field(description="The URL of the deal, as provided in the input")


class DealSelection(BaseModel):
    """
    A class to Represent a list of Deals
    """

    deals: List[Deal] = Field(
        description="Your selection of the 5 deals that have the most detailed, high quality description and the most clear price. You should be confident that the price reflects the deal, that it is a good deal, with a clear description"
    )


class Opportunity(BaseModel):
    """
    A class to represent a possible opportunity: a Deal where we estimate
    it should cost more than it's being offered
    """

    deal: Deal
    estimate: float
    discount: float
