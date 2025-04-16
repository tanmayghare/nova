from amadeus import Client, Location, ResponseError

amadeus = Client(
    client_id='N363uci0ESQBv2mEyzAF4vA3AQ8psshF',
    client_secret='g4fA3MbSiVSa46g6'
)

try:
    response = amadeus.reference_data.locations.get(
        keyword='LON',
        subType=Location.AIRPORT
    )    
    print(response.data)
except ResponseError as error:
    print(error)
