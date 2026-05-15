import qrcode
 
url = input("Enter text here: ")
img = qrcode.make(url)
img.save('qrcode.png')
print("qr code created successfully !")
