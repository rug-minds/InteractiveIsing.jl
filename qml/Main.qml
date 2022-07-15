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
  width: canvas.width + 256
  height: canvas.height + 256
  visible: true

  // UPS Counter
  UpdateCounter{
    anchors.left: parent.left
  }



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



  // Whole application in a column
  // Screen stuff above, panels and text under
  ColumnLayout{
    // anchors.fill: parent
    anchors.centerIn: parent
    spacing: 32

    TopPanel{
      Layout.alignment: Qt.AlignHCenter
    }

    // Middle row
    // Brush - Screen - Temperature
    RowLayout{
      spacing: 6
      Layout.alignment: Qt.AlignHCenter

      BrushPanel{
        id: bpanel
        Layout.alignment: Qt.AlignVCenter
      }


      JuliaCanvas{
        Layout.alignment: Qt.AlignCenter
        id: canvas
        // width: 500
        // height: 500
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
