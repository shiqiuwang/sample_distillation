class RAPLFile:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.baseline = []
        self.process = []
        self.recent = 0
        self.num_process_checks =0
        self.process_average = 0
        self.baseline_average = 0

    def set_recent(self,val):
        self.recent = val

    def create_gpu(self,baseline_average, process_average):
        self.baseline_average = baseline_average
        self.process_average = process_average

    def average(self,baseline_checks):
        self.process_average = sum(self.process)/self.num_process_checks
        self.baseline_average = sum(self.baseline)/baseline_checks

    def __repr__(self):
        print(self.name,self.path,self.recent)
