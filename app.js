'use strict';
const fs = require('fs');
//Require
var hanzi = require("hanzi");
//Initiate
hanzi.start();

var decomposition = hanzi.decomposeMany("轟");
console.log(decomposition);
//var data = JSON.stringify(decomposition, null, 2);
//fs.writeFileSync('decom.json', data);
