"""The script is designed to generate the monthly
MHW report in html format.

1. Download the google doc in html format
2. copy the html code directly to `mhw_report_draft.html`
3. have the html template in dir `mhw_report_template`
4. use jinja to templating the new monthly report `mhw_report_new.html`

one step of mhw_report_text_cron.sh in the crontab job 
"""
import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader

def parse_draft():
    """
    The draft is the direct Google Drive document export html file

    """
    # Read the HTML file
    # draft_file = '/Public/chsu/share_mhw/DraftofGlobalMHWForecastDiscussion.html'
    draft_file = '/home/chsu/mhw_portal/mhw_report_draft.html'
    with open(draft_file, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse the HTML content
    html_parsed = BeautifulSoup(html_content, 'html.parser')

    current_paragraph1 = html_parsed.find_all('p')[1].text.replace("\xa0", " ")

    current_paragraph2 = html_parsed.find_all('p')[2].text.replace("\xa0", " ")

    forecast_paragraph1 = html_parsed.find_all('p')[3].text.replace("\xa0", " ")

    list_regions = []

    for li in html_parsed.find_all('li'):
        try:
            region = li.text.split("\xa0- ")[0]
            region_text = li.text.split("\xa0- ")[1].replace("\xa0", " ")
        except IndexError:
            region, region_text = html_parsed.find_all('li')[3].text.split("-",1)


        list_regions.append({
            'region_name':region,
            'region_text':region_text
        })

    return [
        current_paragraph1,
        current_paragraph2,
        forecast_paragraph1,
        list_regions
    ]

def render_html(year=2000,month=6,current=True):
    """
    render the html template with the information from the 
    draft from google doc.

    """
    # output_file = '/home/chsu/MHW/mhw_report_new.html'
    output_file = '/home/chsu/mhw_portal/mhw_report_new.html'

    month_strings = calendar.month_name[1:]

    if current :
        current_datetime = datetime.now()
        year = current_datetime.year
        month = current_datetime.month

    # start date string
    start_year = year
    start_month = month_strings[month-1]
    start_date = datetime(year=year, month=month, day=1)

    # end date string
    end_date = start_date + relativedelta(months=11)
    end_year = end_date.year
    end_month = month_strings[end_date.month-1]

    # get draft content
    replace_text = parse_draft()

    # open template file
    # environment = Environment(
    #     loader=FileSystemLoader("/home/chsu/MHW/mhw_report_template/")
    # )
    environment = Environment(
        loader=FileSystemLoader("/home/chsu/mhw_portal/mhw_report_template/")
    )
    template = environment.get_template("mhw_report_template.html")
    rendered_html = template.render(
        start_year = start_year,
        start_month = start_month,
        end_year = end_year,
        end_month = end_month,
        current_paragraph1 = replace_text[0],
        current_paragraph2 = replace_text[1],
        forecast_paragraph1 = replace_text[2],
        list_regions = replace_text[3]
    )

    with open(output_file, mode="w", encoding="utf-8") as html_file:
        html_file.write(rendered_html)
        print(f"... {output_file} rendered")


if __name__ == "__main__":
    render_html(current=True)
