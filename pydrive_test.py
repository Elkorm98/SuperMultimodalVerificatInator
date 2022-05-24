
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
def find_folder(file_list, foldername):
    for file in file_list:
        if (file['title'] == foldername):
            folder_id = file['id']
            return folder_id

gauth = GoogleAuth()
gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
drive = GoogleDrive(gauth)

# View all folders and file in your Google Drive
fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
folder_id=find_folder(fileList,'Data')
fileList = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
folder_id=find_folder(fileList,'Res')
fileList=fileList = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
for file in fileList:
    if file['title'].find('Gyroscope')!=-1:
        fileG = drive.CreateFile({'id': file['id']})
        fileG.GetContentFile('predict/Pawel_wrist.csv')

    fileT = drive.CreateFile({'id': file['id']})
    fileT.Trash()