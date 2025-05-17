import jwt
import os
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailVerifier:
    def __init__(self):
        self.secret_key = os.getenv('SECRET_KEY', 'your-secret-key')
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.verification_url = os.getenv('VERIFICATION_URL', 'http://localhost:8501')

    def generate_verification_token(self, email):
        payload = {
            'email': email,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def send_verification_email(self, email):
        token = self.generate_verification_token(email)
        verification_link = f"{self.verification_url}?verify=true&token={token}"
        
        msg = MIMEMultipart()
        msg['From'] = self.smtp_username
        msg['To'] = email
        msg['Subject'] = "Verify Your Email"
        
        body = f"""
        Thank you for registering! Please click the link below to verify your email:
        {verification_link}
        
        This link will expire in 24 hours.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False