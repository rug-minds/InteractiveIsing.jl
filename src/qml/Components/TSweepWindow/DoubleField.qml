import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform

// Double textfield
TextField{

    property var low: 0.
    property var high: 100.
    property var prec: 2

    property double val: low
    text: `${low}`

    property var frst: true

    selectByMouse: true

    validator: DoubleValidator {
        locale: "en"
        bottom: low; top: high
        notation: DoubleValidator.StandardNotation
    }

    property var resetVal: false
    onTextChanged: {
        text = `${custDoubleValidator(text, low, high, prec)}`

        if(parseFloat(text) > low){
            val = parseFloat(text).toFixed(prec)
        }
        else{
            val = low
        }
    }

    // onFocusChanged: {
    //     if(parseFloat(text) < low){
    //         text = `${custDoubleValidator(text, low, high, prec)}`
    //     }
    // }

    function custDoubleValidator(text, low, high, prec){
        Number.prototype.countDecimals = function () {
                if(Math.floor(this.valueOf()) === this.valueOf()) return 0;
                return this.toString().split(".")[1].length || 0; 
        }

        var newText = ""
        var def = Math.max(0.0, low)

        // If textfield is cleared, enter in default value
        if(isNaN(parseFloat(text))){
            newText = `${def}`
            resetVal = true
            return newText
        }

        if(parseFloat(text) > high){
            newText = `${high}`
            return newText
        }

        if(parseFloat(text).countDecimals() > prec){
            newText = `${parseFloat(text).toFixed(prec)}`
            return newText
        }

        if(resetVal == true){
            selectAll()
            resetVal = false
        }

        frst = false

        return text
    }

}