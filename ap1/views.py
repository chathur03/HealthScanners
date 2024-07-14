import json
import requests
from django.shortcuts import render, redirect,get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import Scan
from .mlmodels import *

@csrf_exempt
def home(request):
    if request.method == 'POST' and request.FILES.get('scan'):
        scan = Scan(image=request.FILES['scan'])
        scan.save()

        return redirect(results, scan_id = scan.id)
    return render(request, 'home.html')

def bw_image(request, pk):
    scan = Scan.objects.get(pk=pk)
    # Convert to black and white
    from PIL import Image, ImageOps
    img = Image.open(scan.image.path)
    bw = ImageOps.grayscale(img)
    bw.save(scan.image.path)
    return render(request, 'bw_image.html', {'scan': scan})

@csrf_exempt
def chat(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        message = data.get('message', '')

        api_key = "AIzaSyAyr7vovEdSIPLK43soiSvtHzDAC-mG-UY"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
        headers = {
            'Content-Type': 'application/json',
        }
        payload = json.dumps({
            "contents": [{
                "parts": [{
                    "text": message
                }]
            }]
        })
        response = requests.post(url, headers=headers, data=payload)
        response_data = response.json()

        if response.status_code == 200:
            generated_text = response_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Sorry, I did not understand that.')
        else:
            generated_text = 'Sorry, something went wrong. Please try again.'

        return JsonResponse({'response': generated_text})

    return JsonResponse({'response': 'Invalid request method.'}, status=405)

def results(request, scan_id):
    scan = get_object_or_404(Scan, id=scan_id)
    y_pred = predict_lungD(scan.image.path)
    print(y_pred)
    return render(request, "result_page.html", {"pred" : y_pred})