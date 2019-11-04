import glob
from shutil import copyfile
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] # Define a ordem das emoções
participants = glob.glob("source_emotion/*") #Returns a list of all folders with participant numbers

# Para cada indivíduo faça...
for x in participants:
    part = "%s" % x[-4:] # Armazenando o numero do indivíduo.
    for sessions in glob.glob("%s/*" %x): # Listando os subdiretorios do indivíduo.
        for files in glob.glob("%s/*" %sessions):
            current_session = files[20:-30]
            file = open(files, 'r')
            emotion = int(float(file.readline())) # fazendo a leitura do rótulos nos txts.

            images_file = sorted(glob.glob("source_images/%s/%s/*" %(part, current_session)))
            sourcefile_emotion = images_file[-1] # Capturando path das imagens
            sourcefile_neutral = images_file[0] # O mesmo para imagens neutras
            
            dest_neut = "sorted_set/neutral/%s" % sourcefile_neutral[23:] # Gerando diretório destino para imagens neutras.
            dest_emot = "sorted_set/%s/%s" % (emotions[emotion], sourcefile_emotion[23:]) # O mesmo para as imagens com rótulos diferentes.
            
            copyfile(sourcefile_neutral, dest_neut) # Copiando arquivo para o diretorio destino.
            copyfile(sourcefile_emotion, dest_emot)

