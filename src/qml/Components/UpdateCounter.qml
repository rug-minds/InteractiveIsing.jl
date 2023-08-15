// Updates per frame

// UPF Counter

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform

Item{
  width: childrenRect.width
  height: childrenRect.height
  ColumnLayout{
    Layout.alignment: Qt.AlignHCenter
    Text{
      Layout.alignment: Qt.AlignHCenter
      text: "Updates per frame: "

    }
    Text{
      Layout.alignment: Qt.AlignHCenter
      text: obs.upf.toFixed(3)
    }

    Text{
      Layout.alignment: Qt.AlignHCenter
      text: "Updates per frame per unit: "

    }
    Text{
      Layout.alignment: Qt.AlignHCenter
      text: obs.upfps.toFixed(3)
    }
  }
}