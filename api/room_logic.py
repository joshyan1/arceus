rooms = {}

def get_rooms():
    return list(rooms.keys())

def create_room(room_name):
    if room_name in rooms:
        return False
    rooms[room_name] = []
    return True

def join_room(room_name, device_id):
    if room_name not in rooms:
        return False
    rooms[room_name].append(device_id)
    return True