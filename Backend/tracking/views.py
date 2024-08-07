from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import cv2
# from .yolo import process_frame
from .segment import process_frame
import os
import matplotlib.pyplot as plt
from django.conf import settings
import numpy as np
import matplotlib.patches as mpatches 
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
@csrf_exempt
def index(request):
    processed_image_url=""
    processed_video_url=""
    if request.method == 'POST':
        if 'video' in request.FILES:
            video = request.FILES['video']
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
            filename = fs.save(video.name, video)
            video_path = fs.path(filename)
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)

            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Define the codec and create VideoWriter object
            output_filename = os.path.splitext(filename)[0] + '_processed.avi'
            output_static_path = os.path.join(settings.STATICFILES_DIRS[0],output_filename)
            # print(output_static_path)
            # return render(request, 'tracking/results.html')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec for MP4
            out = cv2.VideoWriter(output_static_path, fourcc, fps, (frame_width, frame_height))
            inferences = []
            postprocess = []
            preprocess = []
            ignoreCount = 20
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                try:

                    processed_frame,results = process_frame(frame)
                    # if len(results) >= 1 :
                        # if ignoreCount <= 0  :
                            # inferences.append(results.speed["inference"])
                            # preprocess.append(results.speed["preprocess"])
                            # postprocess.append(results.speed["postprocess"])
                        # else:
                            # ignoreCount-=1
                    # Write the processed frame to the video
                except:
                    print("error")
                out.write(processed_frame)
            
            # Release everything if job is finished
            cap.release()
            out.release()
            # Provide the processed video for display
            processed_video_url = settings.STATIC_URL + output_filename
            print(output_static_path)
            print(processed_video_url)
            print(out)

            plt.rcParams["figure.figsize"] = [10, 7.5]
            plt.rcParams["figure.autolayout"] = True
            
            # index = range(1,len(preprocess)+1)

            # plt.title("YOLOv10")
            # plt.plot(index,preprocess, color="red")
            # plt.plot(index,inferences, color="green")
            # plt.plot(index,postprocess, color="blue")
            # plt.legend(handles=[
            #     mpatches.Patch(color='blue', label='postprocess'),
            #     mpatches.Patch(color='red', label='preprocess'),
            #     mpatches.Patch(color='green', label='inference')
            # ])
            # plt.axhline(linewidth=1,linestyle="--",color='b',y=np.average(postprocess))
            # plt.axhline(linewidth=1,linestyle="--",color='r',y=np.average(preprocess))
            # plt.axhline(linewidth=1,linestyle="--",color='g',y=np.average(inferences))
            # plt.savefig(output_filename+".jpg")
        if 'photo' in request.FILES:
            photo = request.FILES['photo']
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
            filename = fs.save(photo.name, photo)
            photo_path = fs.path(filename)
            processed_frame,results = process_frame(cv2.imread(photo_path))
            output_filename = os.path.splitext(filename)[0] + '_processed.jpg'
            output_static_path = os.path.join(settings.STATICFILES_DIRS[0],output_filename)
            processed_image_url =settings.STATIC_URL + output_filename
            cv2.imwrite(output_static_path,processed_frame)
        
        return JsonResponse({'video_url': processed_video_url,'image_url':processed_image_url})
        return render(request, 'tracking/results.html', )
    
    return render(request, 'tracking/html2.html')