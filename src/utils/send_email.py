import smtplib
from email.mime.text import MIMEText


def send_email(title, content):
    mail_host = "smtp.qq.com"
    mail_user = "1585765217"
    mail_password = "rdlyvcnyhopyghae"
    sender = "1585765217@qq.com"
    receivers = ["1585765217@qq.com"]

    message = MIMEText(content, "plain", "utf-8")
    message["Subject"] = title
    message["From"] = sender
    message["To"] = receivers[0]

    try:
        smtp = smtplib.SMTP_SSL(mail_host)
        smtp.login(mail_user, mail_password)
        smtp.sendmail(sender, receivers, message.as_string())

        smtp.quit()
        print("SMTP Success")
    except smtplib.SMTPException as e:
        print("SMTP Error:", e)


if __name__ == '__main__':
    title = "test hello smtp"
    content = "test hello smtp"
    send_email(title, content)
