# Parte prática da prova 2 do módulo 6 de Engenharia de Computação

O código deve ser rodado do diretório "exemplos" para funcionar. Foi utilizado como base o repósitorio fornecido pelo Nicola. Utilizei um modelo pronto do YOLO de face recognition. Primeiramente utilizamos o cv2.VideoCapture para pegar os frames do vídeo e o ret(booleano que indica se estamos conseguindo pegar o video ). Após isso, carregamos o modelo e botamos o frame nele para obtermos um resultado. Após isso escrevemos o result[0].plot() no vídeo out.mp4. Esse procedimento ocorre até a variável ret virar falsa o que ocorre no final do código. Além disso, usamos cv2.imshow() para mostrar o vídeo enquanto o modelo atua.
