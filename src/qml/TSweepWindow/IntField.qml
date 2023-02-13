import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform

// Int textfield
TextField{

    property var low: 0
    property var high: 100

    property int val: low
    text: `${low}`

    property var frst: true


    validator: IntValidator {
        bottom: low; top: high
    }
    selectByMouse: true

    property var resetVal: false
    onTextChanged: {
        text = custIntValidator(text, low, high)
        val = parseInt(text)
    }

    function custIntValidator(text, low, high){
        var newText = ""
        var def = Math.max(0, low)

        if(isNaN(parseInt(text))){
            var newInt = def
            newText = `${newInt}`
            resetVal = true
            return newText
        }

        if(parseInt(text) < low){
            newText = `${low}`
            return newText
        }
        elseif(parseInt(text) > high){
            newText = `${high}`
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