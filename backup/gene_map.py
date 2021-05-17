import requests

BASE_URL = "https://maps.googleapis.com/maps/api/staticmap?"
API_KEY = "AIzaSyCxKwvN3fT_vjTG_eqTskLOCnSwlahpuTo"
CITY = "Tainan"
ZOOM = 14

URL = BASE_URL + "center=" + CITY + "&zoom=" + str(ZOOM) + "&size=500x500&key=" + API_KEY
response = requests.get(URL)
with open("tainan.jpeg", "wb") as file:
    print(response.content)
    file.write(response.content)
