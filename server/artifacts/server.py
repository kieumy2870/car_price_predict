from flask import Flask, request, jsonify
import util
app=Flask(__name__)
@app.route('/get_brand_gearbox_origin_type', methods=['GET'])
def feature_name():
    response=jsonify({'brand': util.get_brand_names(),
                      'gearbox': util.get_gearbox(),
                      'origin': util.get_origin(),
                      'type':util.get_type(),
                      'columns': util.get_columns()
    })
    response.headers.add('Acess-Control-Allow-Origin','*')
    return response
@app.route('/predict_car_price', methods=['GET', 'POST'])
def predict_car_price():
    manufacture_date= float(request.form['manufacture_data'])
    seats= float(request.form['seats'])
    mileage_v2=float(request.form['mileage_v2'])
    color=float(request.form['color'])
    model=float(request.form['model'])
    brand=request.form['brand']
    origin=request.form['origin']
    type=request.form['type']
    gearbox=request.form['gearbox']
    response=jsonify({'price':util.get_predict_car_price(manufacture_date, seats, mileage_v2, color, model, brand, origin,type,gearbox)})
    response.headers.add('Acess-Control-Allow-Origin','*')
    return response
if __name__=='__main__':
    print('Starting Python Flask server for prediction...')
    util.load_saved_artifacts()
    app.run()