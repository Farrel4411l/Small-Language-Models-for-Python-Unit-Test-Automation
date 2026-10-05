import urllib.request, urllib.parse, re, ssl
url = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote('site:jdih.kemenkeu.go.id ext:pdf 101/PMK.010/2016')
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req, context=ctx).read().decode('utf-8')
    print("PMK 101:")
    print(re.findall(r'https://jdih\.kemenkeu\.go\.id/download/[^\s\"\'\>]*\.pdf', html))
except Exception as e:
    print(e)

url2 = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote('site:jdih.kemenkeu.go.id ext:pdf PMK 105 Tahun 2025')
req2 = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html2 = urllib.request.urlopen(req2, context=ctx).read().decode('utf-8')
    print("PMK 105:")
    print(re.findall(r'https://jdih\.kemenkeu\.go\.id/download/[^\s\"\'\>]*\.pdf', html2))
except Exception as e:
    print(e)
