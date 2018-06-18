var fs = require('fs');
//Require
var hanzi = require("hanzi");
//Initiate
hanzi.start();

var from_filename = process.argv[2];
var to_filename = process.argv[3];
var list = [];

fs.readFile(from_filename, function(err, data) {
    if(err) throw err;
    var lines = data.toString().trim().split("\n");
    for(i in lines){
        var res = hanzi.decomposeMany(lines[i], 2);
        list.push(res);
    }
    var data = JSON.stringify(list, null, 2);
    fs.writeFileSync(to_filename, data);
});
