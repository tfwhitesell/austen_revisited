def read_text(foldername:str, work_id:int):
    '''
    Open text file in read-only mode.
    Input folder and work_id to retrieve contents.
    '''
    with open(f'../data/{foldername}/{work_id}.txt', 'r') as file:
        print(work_id)
        data = file.read()
    
    return data