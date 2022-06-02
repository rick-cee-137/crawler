from requests_html import HTMLSession, AsyncHTMLSession
from pprint import pprint
from bs4 import BeautifulSoup, SoupStrainer
import requests
import re
from urllib.parse import urlparse, urljoin
import phonenumbers
from email_validator import validate_email, EmailNotValidError
import uuid
#import html2csv
import pandas as pd
import numpy as np
from collections import ChainMap, OrderedDict, namedtuple
import json
from sutime import SUTime
from colorama import init, Fore, Back, Style  # use init for term
import csv
# import requests_cache
from datetime import timedelta


class scraper():
    def __init__(self, url, js_toggle=False):
        self.url = url
        self.js_toggle = False
        self.country_code = 'US'
        self.internal_urls = set()
        self.external_urls = set()
        self.email_unclear = set()
        self.emails = set()
        self.phone_numbers = set()
        self.total_urls_visited = 0
        self.knw_graph = ChainMap()
        self.all_forms = ChainMap()
        self.all_tables = []
        self.sutime = SUTime(mark_time_ranges=True, include_range=True)
        self.all_dates = set()

    def __is_valid(self, url):
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    def __get_data(self, soup):
        text = soup.get_text()
        lines = [line.strip() for line in text.splitlines()]

        lines_list = []
        for i in range(len(lines)):
            if lines[i]:
                lines_list.append(lines[i])
        p = []
        for para in soup.find_all('p'):
            p.append(para.getText().strip())

        data = OrderedDict(
            url=self.url,
            lines=lines_list,
            para_list=p,
            parsed_dates=self.parse_datetime("_".join(p), parser=self.sutime)
        )

        return data

    def get_all_website_links(self, url):
        urls = set()
        self.domain_name = urlparse(url).netloc
        soup = BeautifulSoup(requests.get(url).content, "html.parser")

        if soup.html:
            self.parse_forms(soup)
            self.parse_email_address(soup)
            self.parse_phone_numbers(soup)
            self.parse_tables(soup)

        for a_tag in soup.findAll("a"):
            href = a_tag.attrs.get("href")
            if href == "" or href is None:
                continue
            href = urljoin(url, href)
            parsed_href = urlparse(href)
            if not self.__is_valid(href):
                continue
            if href in self.internal_urls:
                continue
            if self.domain_name not in href:
                if href not in self.external_urls:
                    self.external_urls.add(href)
                continue
            urls.add(href)
            self.internal_urls.add(href)
        if not "Cloudflare" in soup.get_text() or soup.get_text():  # replace with list of cdn providers
            data = self.__get_data(soup)
        else:
            data = []

        return urls, data

    def crawl(self, max_urls):
        self.total_urls_visited += 1
        links, data = self.get_all_website_links(self.url)
        if data:
            self.knw_graph = self.knw_graph.new_child(data)
        # print(Fore.YELLOW + f"[*] fetching {self.url}")
        # print(Style.RESET_ALL)
        for link in links:
            # for link in links:
            if self.total_urls_visited > max_urls:
                break
            self.url = link
            self.crawl(max_urls=max_urls)

    def __get_form_details(self, form):

        details = {}
        action = form.attrs.get("action").lower()
        method = form.attrs.get("method", "get").lower()
        inputs = []
        for input_tag in form.find_all("input"):
            input_type = input_tag.attrs.get("type", "text")
            input_name = input_tag.attrs.get("name")
            input_value = input_tag.attrs.get("value", "")
            inputs.append(
                {"type": input_type, "name": input_name, "value": input_value})

        for select in form.find_all("select"):
            select_name = select.attrs.get("name")
            select_type = "select"
            select_options = []
            select_default_value = ""
            for select_option in select.find_all("option"):
                option_value = select_option.attrs.get("value")
                if option_value:
                    select_options.append(option_value)
                    if select_option.attrs.get("selected"):
                        select_default_value = option_value
            if not select_default_value and select_options:
                select_default_value = select_options[0]
            inputs.append({"type": select_type, "name": select_name,
                          "values": select_options, "value": select_default_value})
        for textarea in form.find_all("textarea"):
            textarea_name = textarea.attrs.get("name")
            textarea_type = "textarea"
            textarea_value = textarea.attrs.get("value", "")
            inputs.append(
                {"type": textarea_type, "name": textarea_name, "value": textarea_value})
        details["action"] = action
        details["method"] = method
        details["inputs"] = inputs
        details["url"] = self.url
        return details

    def parse_forms(self, soup):

        forms = soup.find_all("form")

        for i, form in enumerate(forms, start=1):
            form_details = self.__get_form_details(form)

            self.all_forms = self.all_forms.new_child(
                OrderedDict(form_details))

            print(Fore.GREEN + f"[*]found form# {i}")
            print(Style.RESET_ALL)
            name = "./forms/form_"+str(self.url).strip().replace(" ", "_").replace(
                "https://www.", "").replace("/", "_")+"_"+str(i)+"_.txt"
            with open(name, "w") as text_file:
                text_file.write(str(form_details))

    def parse_email_address(self, soup):
        print(Fore.GREEN+"[*] parsing email_ids")
        print(Style.RESET_ALL)

        EMAIL_REGEX = r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""

        for re_match in re.finditer(EMAIL_REGEX, soup.html.decode()):
            self.email_unclear.add(re_match.group())

    def __check_emails(self):
        print(Fore.GREEN+"[*] verifying email_ids")
        print(Style.RESET_ALL)

        for email in self.email_unclear:
            try:
                self.emails.add(validate_email(
                    email, check_deliverability=True, dns_resolver=None).email)
            except:
                continue

    def parse_phone_numbers(self, soup):
        print(Fore.GREEN+"[*] parsing phone_no's")
        print(Style.RESET_ALL)

        numbers = phonenumbers.PhoneNumberMatcher(
            soup.html.decode(), self.country_code)  # FIXME
        for number in numbers:
            self.phone_numbers.add(number.raw_string)

    def __dump_data(self):
        print(Fore.GREEN+"[*] dumping exctracts")
        print(Style.RESET_ALL)
        with open("./emails_ids_"+str(self.domain_name).strip().replace(" ", "_").replace("https://www.", "").replace("/", "_")+".txt", "w") as text_file:
            text_file.write(str(self.emails))

        with open("./phone_numbers_"+str(self.domain_name).strip().replace(" ", "_").replace("https://www.", "").replace("/", "_")+".txt", "w") as text_file:
            text_file.write(str(self.phone_numbers))

        with open("./datetimes_"+str(self.domain_name).strip().replace(" ", "_").replace("https://www.", "").replace("/", "_")+".txt", "w") as text_file:
            text_file.write(str(self.all_dates))

    def parse_tables(self, soup):
        print(Fore.GREEN+"[*] parsing tables")
        print(Style.RESET_ALL)
        output = []
        for table_num, table in enumerate(soup.find_all('table')):
            name = './tables/table_'+str(self.url).strip().replace(" ", "_").replace(
                "https://www.", "").replace("/", "_")+"_"+str(table_num)+"_.csv"
            csv_string = open(name, 'w')
            csv_writer = csv.writer(csv_string)
            for tr in table.find_all('tr'):
                row = [''.join(cell.stripped_strings)
                       for cell in tr.find_all(['td', 'th'])]
                csv_writer.writerow(row)
            table_attrs = dict(num=table_num)
            csv_string.close()
            try:
                self.all_tables.append(pd.read_csv(name))
            except:
                pass
            output.append((csv_string, table_attrs))
            # self.all_tables.append(output)

    def parse_datetime(self, text_obj, parser):
        print(Fore.GREEN+"[*] parsing datetime")
        print(Style.RESET_ALL)
        start = 0
        end = 0
        preds = parser.parse(text_obj)
        for i in range(len(preds)):
            x = preds[i]
            start = x["start"]
            end = x["end"]
            end = end+20
            start = start-20
            if end > len(text_obj):
                end = len(text_obj)
            if start < 0:
                start = 0
            preds[i]["surround_text"] = text_obj[start:end]
            self.all_dates.add(text_obj[start:end])
        return preds

    def begin(self, max_urls):
        requests_cache.install_cache(
            'http_cache', expire_after=timedelta(days=1))
        self.crawl(max_urls)
        self.__check_emails()
        self.__dump_data()
