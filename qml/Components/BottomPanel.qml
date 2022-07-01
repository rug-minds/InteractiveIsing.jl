import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform

import "../TSweepWindow"

// Panels under plot
// Inserting Defects
// Magnetization
ColumnLayout{
  spacing: 2

  // Defects textfield & Button
  Item{
    Layout.alignment: Qt.AlignCenter

    width: childrenRect.width
    height: childrenRect.height

    ColumnLayout{
      spacing: 2
      Layout.alignment: Qt.AlignHCenter

      TextField{
        Layout.alignment: Qt.AlignHCenter
        id: defectText
        placeholderText: "0"
        validator: IntValidator {
          bottom: 0; top: 100
        }
      }

      Button{
        Layout.alignment: Qt.AlignHCenter
        text: "Insert Defects"
        onClicked: {
          if( !( isNaN(parseInt(defectText.text)) ) ){
          if(parseInt(defectText.text) > 99)
        {
          defectText.text = "99"
        }

        Julia.addRandomDefectsQML(parseInt(defectText.text))
      }
      defectText.text = "";
    }
  }
}
}

// Text under plot, magnetization
Item{
  width: childrenRect.width
  height: childrenRect.height
  Layout.alignment: Qt.AlignHCenter
  ColumnLayout{
    Layout.alignment: Qt.AlignHCenter
    Text{
      Layout.alignment: Qt.AlignHCenter
      text: "Magnetization: " + obs.M.toFixed(1)
    }
  }
}

// Initiate Temperature sweep

Button{
  Layout.alignment: Qt.AlignHCenter
  text: "Temperature Sweep"
  onClicked: {
    var component = Qt.createComponent("../TSweepWindow/Tsweep.qml")
    var window = component.createObject(root)
    window.show()
    //   if(!root.tsweep.active){
    //   println("Active")
    // }

  }
}
}