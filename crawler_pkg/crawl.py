def run(**kwargs):
    from crawler_pkg.helpers import scraper
    # from colorama import init
    import pkg_resources
    import os

    # DATA_PATH = pkg_resources.resource_filename("crawler_pkg", "data/")
    # init(convert=True)
    try:
        if not os.path.isfile("./forms"):
            os.makedirs("./forms")
            os.makedirs("./tables")
    except FileExistsError as f:
        pass

    s = scraper(kwargs.get("url"))
    s.begin(kwargs.get("max_urls"))
#
# if __name__ == "__main__":
#     run(url="https://en.wikipedia.org/wiki/Renewable_energy_in_India#Global_rank", max_urls=10)
