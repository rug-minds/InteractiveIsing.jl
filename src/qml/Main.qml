import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform

import "Components"
import "TSweepWindow"

ApplicationWindow {
  id: root
  title: "Ising Simulation"
  width: canvas.width + 300
  height: canvas.height + 300
  // width: 712
  // height: 712
  visible: true

  // UPS Counter
  UpdateCounter{
    anchors.left: parent.left
  }

  // Save Button
  Image{
    anchors.right: parent.right
    width: 32
    height: 26
    source: "Icons/cam.png"
    MouseArea {
      anchors.fill: parent
      cursorShape: Qt.PointingHandCursor
      hoverEnabled: false
      onClicked: {
        Julia.saveGImgQML()
      }
    }
  }

  TopPanel{
    anchors.top: parent.top
    anchors.horizontalCenter: parent.horizontalCenter
    
      // Layout.alignment: Qt.AlignHCenter
  }

  // Whole application in a column
  // Screen stuff above, panels and text under
  ColumnLayout{
    // anchors.fill: parent
    anchors.centerIn: parent
    spacing: 2

    LayerSelector{
        Layout.alignment: Qt.AlignHCenter
        // anchors.bottom: mainrow.top
    }

    // Middle row
    // Brush - Screen - Temperature
    RowLayout{
      id: mainrow
      spacing: 6
      Layout.alignment: Qt.AlignHCenter

      BrushPanel{
        id: bpanel
        Layout.alignment: Qt.AlignVCenter
      }


      JuliaCanvas{
        Layout.alignment: Qt.AlignCenter
        id: canvas
        // height: 500
        // width: 500
        width: {
          if(obs.gSize > 500)
          {
            return obs.gSize
          }
          else
          {
            return 500
          }
        }
        height: {
          if(obs.gSize > 500)
          {
            return obs.gSize
          }
          else
          {
            return 500
          }
        }

        paintFunction: showlatest

        MouseArea{
          anchors.fill: parent
          onClicked: {
            Julia.circleToStateQML(mouseY, mouseX, bpanel.clamp)
          }
        }

      }

      TempSlider{
        Layout.alignment: Qt.AlignCenter
      }
    }

    BottomPanel{
      Layout.alignment: Qt.AlignCenter
    }
  }


  // Timer for display
  Timer {
    // Set interval in ms:
    property int frames: 1
    interval: 1/60*1000; running: true; repeat: true
    onTriggered: {
      Julia.timedFunctions();
      canvas.update();
      frames += 1
    }
  }
}
