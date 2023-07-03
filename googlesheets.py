
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def insert_row_to_sheet(spreadsheet_id, data):
    # Use credentials to create a client to interact with the Google Drive API
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('path/to/credentials.json', scope)
    client = gspread.authorize(creds)
    
    # Find a workbook by name and open the first sheet
    # Make sure you use the right name here.
    sheet = client.open_by_key(spreadsheet_id).sheet1
    
    # Insert the data
    sheet.append_row(data)

# Specify the ID of your Google Sheet and the data you want to insert
spreadsheet_id = '<your-spreadsheet-id>'
data = ['This', 'is', 'a', 'new', 'row']

insert_row_to_sheet(spreadsheet_id, data)
