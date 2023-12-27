# Kodas recognitionui ir TensorFlow Lite modelis
- Recognition.py – failas su python kodu, kuriame bandau pasitelkęs opencv-python(cv2) biblioteką išsikviesti kamerą ir siųsti frame'us (kadrus) į modulį.
    - Iš modulio nepavyksta gauti jokių detectionų. Neįsivaizduojų kodėl.
- best-fp16_2.tfilte – TensorFlow Lite modulis, kuris buvo konvertuotas iš YOLOv5s modulio.
    - Testuotas Google Colab'e, paduodant jam vieną foto. 
    ![Vidutinio greicio matuoklio zenklas](vidutinis.jpg)
    - Detection'as veikia puikiai, tą patį reikėtų gauti Recognition.py faile, naudojant kamerą.