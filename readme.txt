Csharp_kehuduan.py: client
flask-video-streaming-master/app.py :server
-------
this version  Put the two modules  together between camera_opencv.py and app.py.
because the global start x cound't transform to the camera_opencv.py, even though I try to 
set  the start x be global ,and try to use the "from app import start x" , but it didn't wok will.
so,this version can be tested well, but the detection should transform the counts of the boxes 
and the image together,it will be done tomorrow.
------- 
run the python call.py start the server 
