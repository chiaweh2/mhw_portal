import datetime
import subprocess
import sys
from bs4 import BeautifulSoup


def construct_body(soup):
    """
    Contructing the body of the email based on 
    the set format of marine heatwave report page.

    If the structure of the page change the
    code in this function might need to be adjusted.
    """
    # Find a title
    title = soup.find('h2')  # Example: Extract text within a <p> tag
    title = f"<h2> {title.text} </h2>"

    # google form
    googleform = (
        '<p> Please feel free to provide feedback and comments through '+
        'https://forms.gle/P9B33TYueo97C3ts8</p>'
    )

    # Find banner
    banner1 = soup.find('div', class_='alert-secondary')
    banner1 = '<p> ('+banner1.text.strip() + ') <p>'

    banner2 = soup.find('div', class_='alert-warning')
    banner2 = '<p> ('+banner2.text.strip() + ') <p>'

    # Find a forecast time
    forecast_initime = f"<p> {soup.find_all('h5')[0].text} </p>"
    forecast_period = f"<p> {soup.find_all('h5')[1].text} </p>"
    add_link = (
        '<p> For details and a more interactive interface, '+
        'please visit https://psl.noaa.gov/marine-heatwaves/#report </p>'
    )

    # parsed subtitle and content in order
    subtitle1 = f"<h3> [{soup.find_all('h3')[0].text}] </h3>"
    content1 = f"<p> {soup.find('p', class_='smallP').text.strip()} </p>"

    subtitle2 = f"<h4> {soup.find_all('h4')[0].text} </h4>"
    # without warning new method banner
    content21 = f"<p> {soup.find_all('p')[1].text.strip()} </p>"
    content22 = f"<p> {soup.find_all('p')[2].text.strip()} </p>"
    # with warning new method banner
    # content21 = f"<p> {soup.find_all('p')[3].text.strip()} </p>"
    # content22 = f"<p> {soup.find_all('p')[4].text.strip()} </p>"

    subtitle3 = f"<h4> {soup.find_all('h4')[1].text} </h4>"
    # without warning new method banner
    content3 = f"{soup.find_all('p')[3]}"
    # with warning new method banner
    # content3 = f"{soup.find_all('p')[5]}"

    all_content = (
        "Content-Type: text/html\n\n<html><body>"+
        title+
        googleform+
        banner1+
        banner2+
        forecast_initime+
        forecast_period+
        add_link+
        "<p>===========</p>"+
        subtitle1+
        content1+
        "<p>---</p>"+
        subtitle2+
        content21+
        content22+
        "<p>---</p>"+
        subtitle3+
        content3+
        "</body></html>"
    )

    return all_content

# def construct_html_body(soup):
#     """
#     Contructing the body of the email based on 
#     the set format of marine heatwave report page.

#     If the structure of the page change the
#     code in this function might need to be adjusted.
#     """
#     print('test')

def send_email(recip_email, sub, body):
    """
    The function send the email out using the 
    linux/unix sendmail function
    """

    sendemail = f'echo -e "To: {recip_email}\nSubject: {sub}\n{body}" | sendmail -t'

    # execute send email
    subprocess.call(
        sendemail,
        shell=True,
        executable="/usr/bin/bash"
        )


if __name__ == "__main__":

    # get the date when the email be generated
    date = datetime.date.today()
    try:
        if sys.argv[1] == 'mlist':
            RECIPIENT_EMAIL = 'psl.mhw.forecast@list.woc.noaa.gov'
        elif sys.argv[1] == 'test':
            RECIPIENT_EMAIL = 'chia-wei.hsu@noaa.gov'
        else :
            RECIPIENT_EMAIL = None
    except IndexError:
        sys.exit('please put argument value of "mlist" or "test"')

    SUBJECT = f'NEW!!! {date.strftime("%Y %B")} Marine Heatwave Forecast Discussion'

    # Read the HTML file
    with open('/httpd-test/psd/marine-heatwaves/report.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse the HTML content
    html_parsed = BeautifulSoup(html_content, 'html.parser')
    BODY = construct_body(html_parsed)

    # send out the email
    if RECIPIENT_EMAIL:
        send_email(RECIPIENT_EMAIL, SUBJECT, BODY)
