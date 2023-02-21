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
  minimumWidth: ((obs.qmlwidth < 500) ? 500 : obs.qmlwidth) + 300
  minimumHeight: ((obs.qmlwidth < 500) ? 500 : obs.qmlwidth) + 300
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
  // Column{
  //   id: midcolumn
  //   // anchors.fill: parent
  //   anchors.centerIn: parent
  //   spacing: 2

  LayerSelector{
    // Layout.alignment: Qt.AlignHCenter
    // anchors.bottom: mainrow.top
    anchors.horizontalCenter: canvas.horizontalCenter
    anchors.bottom: canvas.top
  }

  // Middle row
  // Brush - Screen - Temperature
  // RowLayout{
  // Row{
  // id: mainrow
  // spacing: 6
  // Layout.alignment: Qt.AlignHCenter

  Item{
    anchors {
      left: root.left
      verticalCenter: canvas.verticalCenter
      leftMargin: 10
      // rightMargin: 10
    }
    width: (root.width - canvas.width)/2
    height: canvas.height



    BrushPanel{
      id: bpanel
      anchors.centerIn: parent
    }
  }


  Item{
    id: canvas
    anchors.centerIn: parent
    height: (obs.qmlwidth < 500) ? 500 : obs.qmlwidth
    width: (obs.qmlwidth < 500) ? 500 : obs.qmlwidth
    JuliaCanvas{
      // Layout.alignment: Qt.AlignCenter
      anchors.centerIn: parent
      // id: canvas
      id: jlcanvas

      width: obs.qmlwidth
      height: obs.qmllength

      paintFunction: showlatest

      MouseArea{
        anchors.fill: parent
        onClicked: {
          Julia.circleToStateQML(mouseY, mouseX, bpanel.clamp)
        }
      }

    }
  }

  Item{
    anchors.right: parent.right
    anchors.verticalCenter: canvas.verticalCenter
    // anchors {
    //   right: root.right
    //   verticalCenter: canvas.verticalCenter
    //   rightMargin: 10
    // }

    width: (root.width - canvas.width) / 2
    height: canvas.height

    TempSlider{
      anchors.centerIn: parent
    }
  }

  // }

  BottomPanel{
    anchors.bottom: parent.bottom
    anchors.horizontalCenter: parent.horizontalCenter
  }
  // }


  // Timer for display
  Timer {
    // Set interval in ms:
    property int frames: 1
      interval: 1/60*1000; running: true; repeat: true
      onTriggered: {
        Julia.timedFunctionsQML();
        jlcanvas.update();
        frames += 1
      }
    }
  }
