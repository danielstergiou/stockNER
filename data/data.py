import labelbox as lb

client = lb.Client(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbTV3NGJlMmcwYmxjMDcwaGY3OWJhdXR1Iiwib3JnYW5pemF0aW9uSWQiOiJjbTV3NGJlMjgwYmxiMDcwaGdnOW1mMzh4IiwiYXBpS2V5SWQiOiJjbTV3ODNzejgwZWo1MDd4MDk5M3kwdDduIiwic2VjcmV0IjoiOWNhNDlkOTE0ZTI1NjI2ODc3YWFlYWRkMjMzYzIyZDQiLCJpYXQiOjE3MzY4NDQwNTMsImV4cCI6MTczOTQzNjA1M30.5Nh1Nw8emc0AnlFzwHOzP5kaHOaiX9BBWLYVTMdasHA")
dataset = client.create_dataset(name="WallStreetBets5000")

payloads = []
with open('Datasets/dataset5000.txt', 'r') as file:
    for line in file:
        payloads.append({
            "row_data": line.strip(),
            "global_key": f"text_{len(payloads)}"
        })

task = dataset.create_data_rows(payloads)
task.wait_till_done()
print(task.errors)