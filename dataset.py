# create an interface function to get splited dataset
# normalization could be added later

def getMFCCDataset():

    MFCCs_DATA = "MFCCsData"
    numpy_datas = []

    dirlist = os.listdir(MFCCs_DATA)
    for d in dirlist:
        d = os.path.join(MFCCs_DATA, d)
        datalist = os.listdir(d)
        datalist = [[np.load(os.path.join(d,x)), os.path.join(d,x)] for x in datalist]
        numpy_datas.extend(datalist)

    for i in range(len(numpy_datas)):
        numpy_datas[i][0] = np.resize(numpy_datas[i][0], (19,512))

    collection = {}
    angry = []
    happy = []
    normal = []

    for i in range(len(numpy_datas)):
        file_name = numpy_datas[i][1]
        if "angry" in file_name:
            numpy_datas[i][1] = np.array([1,0,0])
            angry.append(numpy_datas[i])
        elif "happy" in file_name:
            numpy_datas[i][1] = np.array([0,1,0])
            happy.append(numpy_datas[i])
        else:
            numpy_datas[i][1] = np.array([0,0,1])
            normal.append(numpy_datas[i])

    random.shuffle(angry)
    random.shuffle(happy)
    random.shuffle(normal)

    train_data = angry[:int(len(angry)*0.6)] + happy[:int(len(happy)*0.6)] + normal[:int(len(normal)*0.6)]
    valid_data = angry[int(len(angry)*0.6):int(len(angry)*(0.6+0.2))] + happy[int(len(happy)*0.6):int(len(happy)*(0.6+0.2))] + normal[int(len(normal)*0.6):int(len(normal)*(0.6+0.2))]
    test_data = angry[int(len(angry)*(0.6+0.2)):] + happy[int(len(happy)*(0.6+0.2)):] + normal[int(len(normal)*(0.6+0.2)):]
    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)

    collection['X_train'], collection['Y_train'] = data2vector(train_data)
    collection['X_valid'], collection['Y_valid'] = data2vector(valid_data)
    collection['X_test'], collection['Y_test'] = data2vector(test_data)

    # further add normalization
    return collection