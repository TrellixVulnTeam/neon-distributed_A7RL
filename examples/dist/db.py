from influxdb import InfluxDBClient
import time
import os

class DBServer():
    def start_service(self, db_name='default'):
        os.system('service influxdb start')
        #self.client = InfluxDBClient('localhost', 8086, 'root', 'root', 'default')
	#self.client.create_database(db_name)
 
    def stop_service(self):
        os.system('service influxdb stop')

class DBClient():
    def __init__(self, host, port, db_name='default', worker_name='test', worker_id=-1, measure='neon'):
        self.db_name     = db_name
        self.worker_name = worker_name
	self.worker_id   = worker_id
        self.measure     = measure

	self.client      = InfluxDBClient(host, 8086, 'root', 'root', db_name)
	self.client.create_database(db_name)

	#self.client.switch_database(worker_name)
        self.cache = []

    def set_worker(self, worker_name, worker_id):
        self.worker_name = worker_name
        self.worker_id = worker_id

    def clear_cache(self):
	#print "cache cleared"
        self.cache = []

    def commit_cache(self, event, action, epoch=-1, batch=-1):
	json_body = {
            "measurement": self.measure,
            "tags": {
                #"host": "server01",
                #"region": "us-west"
            },
            "time": long(time.time()*1000),
            "fields": {
                "event": event,
                "action": action,
                "worker_name": self.worker_name,
		"worker_id": self.worker_id,
                "epoch": epoch,
		"batch": batch
            }
        }
	#print "committed"
	self.cache.append(json_body)

    def push_data(self):
	self.client.write_points(self.cache)
	#print "data pushed"
	self.cache = []
	return True

    def query_server(self, server_id):
        #data = self.client.query("SELECT * FROM WHERE server_id = " + server_id)
        data = self.client.query("SELECT * FROM " + self.measure)
	return data

    def clear_data(self):
        try:
            self.client.query("DROP MEASUREMENT " + self.measure)
        except:
            print 'No such measurement: ' + self.measure 
            pass
